import numpy as np
from scipy.fft import rfft, irfft
from statistics import mode


def chunk_and_mask(est, gt, sr, moving_frame_width_ms = 250,
                   rms_threshold=1e-3):
    """
    Splits arrays into chunks/frames along the time dimension, discards chunks with low power and applies a hanning window to each chunk.
    input: (*, 2, T), (*, 2, T)
    
    Returns: (*, C, FW)
    
    C: Number of chunks
    T: Time samples
    FW: Frame width
    """
    
    FW = int(round(1e-3 * moving_frame_width_ms * sr))

    # Total number of chunks
    C = 1 + (gt.shape[-1] - 1)// FW

    # Drop samples to get a multiple of frame size
    if gt.shape[-1] % FW != 0:
        pad_amount = FW-(gt.shape[-1]%FW)
        pad = np.zeros((*gt.shape[:-1], pad_amount))
        gt = np.concatenate([gt, pad], axis=-1)
        est = np.concatenate([est, pad], axis=-1)
    
    assert gt.shape[-1] % FW == 0
    assert est.shape[-1] % FW == 0
    
    # Split signals into frames
    gt = np.array(np.split(gt, C, axis=-1)) # (C, *, 2, FW)
    est = np.array(np.split(est, C, axis=-1))
    
    # Get mask for all chunks based on rms value on either channel
    chunk_rms = np.sqrt(np.mean(gt ** 2, axis=-1)).max(axis=-1)
    chunk_mask = chunk_rms >= rms_threshold
    
    return est, gt, chunk_mask

def compute_ild(s_left, s_right):
    sum_sq_left = np.sum(s_left ** 2, axis=-1)
    sum_sq_right = np.sum(s_right ** 2, axis=-1)
    return 10 * np.log10(sum_sq_left / sum_sq_right)

def ild_diff(s_est, s_gt, sr = None, moving=False):
    """
    Computes the ILD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """

    if moving:
        assert sr is not None, "For moving sources, sr must be given to compute chunks"
        s_est, s_gt, chunk_mask = chunk_and_mask(s_est, s_gt, sr)

    ild_est = compute_ild(s_est[..., 0, :], s_est[..., 1, :])
    ild_gt = compute_ild(s_gt[..., 0, :], s_gt[..., 1, :])

    if moving:
        est_ild_list = []
        gt_ild_list = []
        for i in range(chunk_mask.shape[-1]):
            est_ilds_at_batch = ild_est[chunk_mask[..., i], i]
            gt_ilds_at_batch = ild_gt[chunk_mask[..., i], i]

            est_ild_at_batch = np.mean(est_ilds_at_batch)
            gt_ild_at_batch = np.mean(gt_ilds_at_batch)
        
            est_ild_list.append(est_ild_at_batch)
            gt_ild_list.append(gt_ild_at_batch)

        ild_est = np.array(est_ild_list)
        ild_gt = np.array(gt_ild_list)

    # print("ILD Est", ild_est)
    # print("ILD GT", ild_gt)

    return np.abs(ild_est - ild_gt)

def axiswise_xcorr(a, b, axis=-1, phat=False):
    A = rfft(a, axis=axis)
    B = rfft(b, axis=axis)
    R = A * np.conjugate(B)

    if phat:
        R /= np.abs(R)
    
    ret = irfft(R, axis=axis)
    return ret

def compute_itd(s_left, s_right, sr, t_max = None):
    # corr = signal.correlate(s_left, s_right, axis=)
    corr = axiswise_xcorr(s_left, s_right, axis=-1)

    mid = corr.shape[-1]//2

    # if True:
    if (t_max is None) or t_max > mid:
        t_max = mid
    
    cc = np.concatenate((corr[..., -t_max:], corr[..., :t_max+1]), axis=-1)
    
    # plt.plot(cc[0])
    # plt.scatter([np.argmax(cc[0])], [np.max(cc[0])], color='g')
    # plt.show()

    # plt.plot(cc[0, 0])
    # plt.scatter([np.argmax(cc[0, 0])], [np.max(cc[0, 0])], color='g')
    # plt.show()

    tau = np.argmax(np.abs(cc), axis=-1)
    tau -= t_max

    return tau / sr * 1e6


def itd_diff(s_est, s_gt, sr, moving=False):
    """
    Computes the ITD error between model estimate and ground truth
    input: (*, 2, T), (*, 2, T)
    """
    TMAX = int(round(1e-3 * sr))
    
    if moving:
        s_est, s_gt, chunk_mask = chunk_and_mask(s_est, s_gt, sr)
    
    itd_est = compute_itd(s_est[..., 0, :], s_est[..., 1, :], sr, TMAX)
    itd_gt = compute_itd(s_gt[..., 0, :], s_gt[..., 1, :], sr, TMAX)
    
    # Take the mode at ITD each frame
    if moving:
        itd_diff = np.zeros(chunk_mask.shape[-1])
        for i in range(chunk_mask.shape[-1]):
            est_itds_at_batch = itd_est[chunk_mask[..., i], i]
            gt_itds_at_batch = itd_gt[chunk_mask[..., i], i]
            # print(f'Batch {i} EST ITD', est_itds_at_batch)
            # print(f'Batch {i} GT ITD', gt_itds_at_batch)

            avg_itd_diff = np.mean(np.abs(np.array(est_itds_at_batch) - np.array(gt_itds_at_batch)))
            itd_diff[i] = avg_itd_diff 
    # print("ITD Est", itd_est)
    # print("ITD GT", itd_gt)
    else:
        itd_diff = np.abs(itd_est - itd_gt)
    
    return itd_diff

def test():
    # REQUIRED FOR TESTING ONLY
    import soundfile as sf
    def write_audio_file(file_path, data, sr):
        """
        Writes audio file to system memory.
        @param file_path: Path of the file to write to
        @param data: Audio signal to write (n_channels x n_samples)
        @param sr: Sampling rate
        """
        sf.write(file_path, data.T, sr)

    np.random.seed(0)
    
    sr = 8000
    T = 5
    RSCALE = 0.5

    def get_binaural_chirp(ITD_samples):
        SHIFT = ITD_samples
        t = np.arange(0, T, 1/sr)
        x = np.cos(2 * np.pi * (100 + 250 * t) * t)
        x = np.expand_dims(x, 0)
        y = np.roll(x, -SHIFT) * RSCALE
    
        gt = np.concatenate([x, y], axis=0)
        return gt
    
    ests = []
    gts = []
    for shift in range(-4, 5):
        gt = get_binaural_chirp(shift) * 0.1
        est = gt + np.random.normal(0, 1, size=gt.shape) * 0.1
        
        gts.append(gt)
        ests.append(est)
    
    est = np.array(ests)
    gt = np.array(gts)
    
    write_audio_file('tests/gt.wav', gt[0], sr)
    write_audio_file('tests/est.wav', est[0], sr)
    
    delta_itd = itd_diff(est, gt, sr)#, moving=True)
    # delta_ild = ild_diff(est, gt, sr)#, moving=True)
    
    print(f'Delta ITD: {delta_itd}us')
    # print(f'Delta ILD: {delta_ild}dB')
    
