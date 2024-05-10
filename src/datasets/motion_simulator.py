import ctypes
import numpy as np
import os
import matplotlib.pyplot as plt

import random
import torch
import torchaudio

from src.datasets.multi_ch_simulator import CIPICSimulator

def _plot_coordinates(coords, title):
    x0 = coords
    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(projection='3d')
    ax.scatter(x0[:, 0], x0[:, 1], x0[:, 2])
    ax.set_xlim([-1.0, 1.0])
    ax.set_ylim([-1.0, 1.0])
    ax.set_zlim([-1.0, 1.0])
    ax.set_xlabel('x (m)')
    ax.set_ylabel('y (m)')
    ax.set_title(title)
    plt.show()

def to_float_arr(data: np.ndarray):
    data = data.astype(np.float32).flatten()
    length = len(data)
    return data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(length)

class MotionSimulator():
    def __init__(self, sr: int, frame_duration: float = 0.025) -> None:
        self.lib = ctypes.CDLL('./motion_simulator/moving_sources.so')
        
        self.sr = ctypes.c_int(sr)
        self.frame_duration = ctypes.c_float(frame_duration)

        self.simulator_p = ctypes.c_void_p()
        self._call("simulator_init", ctypes.pointer(self.simulator_p), self.sr,\
                    self.frame_duration, init=True)

    def _call(self, lib_function: str, *args, **kwargs):
        if ('init' in kwargs)  and (kwargs['init'] == True):
            err_code = getattr(self.lib, lib_function)(*args)
        else:
            err_code = getattr(self.lib, lib_function)(self.simulator_p, *args)
        assert err_code == 0, f"Something went wrong during call: {lib_function}"

    def set_hrtf(self, hrtf_file: str):
        self.hrtf_file = ctypes.c_char_p(hrtf_file.encode('utf-8'))

        assert os.path.exists(hrtf_file), f"HRTF not found: {hrtf_file}"
        self._call('simulator_set_hrtf', self.hrtf_file)

    def add_source(self, data: np.ndarray, path: np.ndarray):
        assert (len(path.shape) == 2) and (path.shape[1] == 3),\
                f"Path must have a shape (N, 3), found {path.shape}"

        num_points = path.shape[0]
        num_audio_samples = data.shape[-1]

        expected_time = (num_audio_samples/self.sr.value)
        expected_number_of_frames = int(np.ceil(expected_time / self.frame_duration.value))
        assert num_points >= expected_number_of_frames,\
                f"Number of points in source path must be compatible with audio length "\
                f"and frame duration. Currently, the number of points is {num_points}, but "\
                f"since the audio is {expected_time} seconds long, then with a frame duration "\
                f"of {self.frame_duration.value:.03f}s, the source path must have at "\
                f"least {expected_number_of_frames} points."

        audio, audio_length = to_float_arr(data)
        path, path_length = to_float_arr(path)

        self._call('simulator_add_source', audio, audio_length, path, path_length)

    def simulate(self):
        data_p = ctypes.POINTER(ctypes.c_float)()
        num_sources = ctypes.c_int(0)
        num_samples = ctypes.c_int(0)
        
        self._call("simulator_simulate", ctypes.pointer(data_p),
                   ctypes.pointer(num_sources), ctypes.pointer(num_samples))
        
        num_samples = num_samples.value
        num_sources = num_sources.value
        num_floats = num_samples * num_sources
        
        audio = np.array(data_p[:num_floats], dtype=np.float32).reshape(num_sources, num_samples//2, 2)
        audio = np.transpose(audio, axes=(0, 2, 1))
        
        self._call("simulator_cleanup")

        return audio

    def __del__(self):
        self._call("simulator_destroy")

class CIPICMotionSimulator2(CIPICSimulator):
    def __init__(self, sofa_text_file, sr, 
                 frame_duration = 0.025,
                 use_piecewise_arcs = False) -> None:
        super().__init__(sofa_text_file, sr)

        self.sr = sr
        self.frame_duration = frame_duration
        self._simulator = MotionSimulator(sr, frame_duration)
        self.use_piecewise_arcs = use_piecewise_arcs

    def get_piecewise_arc_path(self, rng: random.Random, t: np.ndarray, debug=False):
        """
        Generates a random path for a source.
        Path must be a numpy array of 3D points, each point being the souce position 
        at some time instance. The array t contains the time values that the simulation
        will use, i.e., the i-th position in the output array will be used as 
        the source position at time t[i].
        """
        # Theta = azimuthal angle
        theta0 = rng.uniform(0, 2 * np.pi)
        
        # Phi = coaltitude angle
        phi0 = rng.uniform(0, np.pi / 2) # So that sources have +ve Z
        
        dt = t[1] - t[0]
        bernoulli_p = dt # Probability for each bernoulli trial

        theta = np.zeros_like(t)
        phi = np.zeros_like(t)
        max_prob = np.ones_like(t) * bernoulli_p
        prob = np.zeros_like(t)
        
        i = 0
        while i < len(t):            
            p = rng.random()
            prob[i] = p
            
            if p < bernoulli_p:
                # User moves!
                
                # Choose number of seconds over which they will
                # perform the motion
                seconds = rng.uniform(0.1, 1.0)
                nsteps = int(round(seconds / dt)) # Equivalent steps

                # Choose speed for theta (positive or negative)
                wtheta = rng.uniform(np.pi / 6, np.pi / 2) * ((-1) ** rng.randint(0, 1))

                # Choose speed for phi (positive or negative)
                wphi = rng.uniform(np.pi / 6, np.pi / 2) * ((-1) ** rng.randint(0, 1))

                j = 0
                while i < len(t) and j < nsteps:
                    theta0 = theta0 + wtheta * dt
                    phi0 = phi0 + wphi * dt
                    
                    theta[i] = theta0
                    phi[i] = phi0
                    i += 1
                    j += 1
            else:
                theta[i] = theta0
                phi[i] = phi0
                i += 1

        path_x = np.sin(phi) * np.cos(theta)
        path_y = np.sin(phi) * np.sin(theta)
        path_z = np.cos(phi) * np.ones_like(t)

        path = np.array([path_x, path_y, path_z]).T # Shape must be (N, 3)
        
        if debug:
            return path, theta, phi, prob, max_prob
        else:
            return path, 0.0

    def get_random_source_path(self, rng: random.Random, t: np.ndarray):
        """
        Generates a random path for a source.
        Path must be a numpy array of 3D points, each point being the souce position 
        at some time instance. The array t contains the time values that the simulation
        will use, i.e., the i-th position in the output array will be used as 
        the source position at time t[i].
        """
        if self.use_piecewise_arcs:
            return self.get_piecewise_arc_path(rng, t)


        # Theta = azimuthal angle
        theta0 = rng.uniform(0, np.pi / 2) # So that sources have +ve Z
        w_theta = rng.uniform(-np.pi / 2, np.pi / 2)

        # Phi = coaltitude angle
        phi0 = rng.uniform(0, np.pi / 2)

        theta_t = theta0 + w_theta * t
        path_x = np.sin(phi0) * np.cos(theta_t)
        path_y = np.sin(phi0) * np.sin(theta_t)
        path_z = np.cos(phi0) * np.ones_like(t)

        path = np.array([path_x, path_y, path_z]).T # Shape must be (N, 3)

        return path, w_theta

    def get_face_to_face_source_path(self, seed: int, t: np.ndarray):
        """
        Generates a path for a source that is always facing the listener following a
        normal distribution around the face to face position with a standard deviation
        of 6 degrees.
        """
        rng = np.random.RandomState(seed)
        path = []
        max_error = rng.uniform(0, np.pi / 10)
        theta_phi = rng.uniform(np.pi/2 - max_error, np.pi/2 + max_error, size=(len(t), 2)) # [*, 2]
        path_x = np.sin(theta_phi[:, 1]) * np.cos(theta_phi[:, 0])
        path_y = np.sin(theta_phi[:, 1]) * np.sin(theta_phi[:, 0])
        path_z = np.cos(theta_phi[:, 1])
        path = np.array([path_x, path_y, path_z]).T # Shape must be (N, 3)
        return path, max_error

    def set_hrtf(self, hrtf_file: str):
        self._simulator.set_hrtf(hrtf_file)

    def simulate(self, srcs, noise, seed=None,
                 face_to_face_idx=None, debug=False):
        """
        Simulate binaural recordings with moving sources from monaural recordings
        using a random HRIR obtained from the CIPIC database. Subject as well as
        HRIR index are randomly chosen based on the seed. The results can be made
        reproducible by setting the seed.

        Args:
            srcs ([np.ndarray]): Monaural sources
            noise (np.ndarray): Monaural noise
            seed (int, optional): Seed for random number generator. Defaults to None.
        """
        # Create array of time steps to define the source paths on
        # Each point in the path is separated by 'frame_duration' seconds
        # Total path duration must be at least as long as the source audio
        simulation_time = srcs[0].shape[-1] / self.sr

        t = np.arange(0, self.frame_duration + simulation_time, self.frame_duration)

        rng = random.Random(seed)

        # Choose HRTF at random
        hrtf = rng.choice(self.sofa_files)
        self._simulator.set_hrtf(hrtf)

        bi_srcs = []
        path_srcs =[]

        params = []
        for i, src in enumerate(srcs):
            if face_to_face_idx is not None and i == face_to_face_idx:
                path, param = self.get_face_to_face_source_path(seed, t)
                # _plot_coordinates(path, f'Enroll')
            else:
                path, param = self.get_random_source_path(rng, t)
                # _plot_coordinates(path, f'Source {i}')

            params.append(param)
            self._simulator.add_source(src, path)
            path_srcs.append(path)
        
        noise_path, param = self.get_random_source_path(rng, t)
        self._simulator.add_source(noise, noise_path)
        
        binaural_sources = self._simulator.simulate()

        bi_srcs, bi_noise = binaural_sources[:-1], binaural_sources[-1]

        # if face_to_face_idx is not None:
        #     from IPython.display import Audio
        #     from IPython.core.display import display
        #     display(Audio(bi_srcs[face_to_face_idx], rate=self.sr))

        if debug:
            return bi_srcs, bi_noise, path_srcs, noise_path
        else:
            return bi_srcs, bi_noise, params

class RRBRIRMotionSimulator(CIPICMotionSimulator2):
    """
    Contains impulse response for [-90,90] deg. azimuth and 0 deg. elevation. at 1.5m.
    """
    def get_random_source_path(self, rng: random.Random, t: np.ndarray):
        """
        Generates a random path for a source. Path must be a numpy array of 3D points,
        each point being the souce position at some time instance. The array t contains
        the time values that the simulation will use, i.e., the i-th position in the
        output array will be used as the source position at time t[i].
        """
        # Theta = azimuthal angle
        theta0 = rng.uniform(-np.pi/2, np.pi/2) # So that sources have +ve Z
        w_theta = rng.uniform(-np.pi/2, np.pi/2)
        r = 1.5

        # Phi = polar angle
        phi0 = np.pi/2

        # Motion in spherical coordinates with fold over at pi/2 and -pi/2
        theta_t = theta0 + w_theta * t

        # Convert to cartesian coordinates
        path_x = np.abs(r * np.sin(phi0) * np.cos(theta_t))
        path_y = r * np.sin(phi0) * np.sin(theta_t)
        path_z = r * np.cos(phi0) * np.ones_like(t)

        path = np.array([path_x, path_y, path_z]).T # Shape must be (N, 3)

        return path

    def get_face_to_face_source_path(self, seed: int, t: np.ndarray):
        """
        Generates a path for a source that is always facing the listener following a
        normal distribution around the face to face position with a standard deviation
        of 6 degrees.
        """
        rng = np.random.RandomState(seed)
        r = 1.5 # m
        theta_phi = rng.multivariate_normal(
            mean=[0, np.pi/2], cov=[[np.pi/30, 0], [0, 0]], size=len(t)) # [*, 2]
        theta_phi[:, 0] = np.clip(theta_phi[:, 0], -np.pi/2, np.pi/2)
        path_x = r * np.sin(theta_phi[:, 1]) * np.cos(theta_phi[:, 0])
        path_y = r * np.sin(theta_phi[:, 1]) * np.sin(theta_phi[:, 0])
        path_z = r * np.cos(theta_phi[:, 1])
        path = np.array([path_x, path_y, path_z]).T # Shape must be (N, 3)
        return path

def test_bindings():
    import librosa
    from scipy.io.wavfile import write as wavwrite
    import time

    def read_audio_file(file_path, sr):
        """
        Reads audio file to system memory.
        """
        return librosa.core.load(file_path, mono=False, sr=sr)[0]

    def read_with_pad(path, fs):
        y = read_audio_file(path, fs)
        if y.shape[-1] < T * fs:
            y = np.pad(y, (0, 0, T * fs - y.shape[-1]))
        else:
            y = y[:T * fs]

        return y

    def write_audio_file(file_path, data, sr):
        """
        Writes audio file to system memory.
        @param file_path: Path of the file to write to
        @param data: Audio signal to write (n_channels x n_samples)
        @param sr: Sampling rate
        """
        wavwrite(file_path, sr, data.T)

    T = 5
    dt = 0.1
    fs = 8000

    theta0 = np.pi/2
    phi0 = 0

    w_theta = np.pi / 4
    w_phi = np.pi

    t = np.arange(0, T, dt)
    rho = np.ones_like(t)
    phi = np.zeros_like(t)
    phi[0] = phi0

    theta = np.zeros_like(t)
    theta[0] = theta0

    # Random walk on theta
    for i in range(1, len(t)):
        sgn_wtheta = 0#1 - 2 * np.random.randint(0, 2)
        sgn_wphi = 1# - 2 * np.random.randint(0, 2)
        
        phi[i] = phi[i-1] + sgn_wphi * w_phi * dt
        theta[i] = theta[i-1] + sgn_wtheta * w_theta * dt
    
    x = np.sin(theta) * np.cos(phi) * rho
    y = np.sin(theta) * np.sin(phi) * rho
    z = np.cos(theta) * rho

    path = np.array([x, y, z]).T

    y1 = read_with_pad('tests/speech_samples/speech1.wav', fs)
    y2 = read_with_pad('tests/speech_samples/speech2.wav', fs)
    y3 = read_with_pad('tests/speech_samples/speech3.wav', fs)
    y4 = read_with_pad('tests/speech_samples/speech4.wav', fs)


    simulator1 = MotionSimulator(fs, dt)
    print('INITIALIZED')
    simulator1.set_hrtf("tests/Test.sofa")
    print('ADDSOURCE')
    simulator1.add_source(y1, path)
    simulator1.add_source(y2, -path)

    simulator2 = MotionSimulator(fs, dt)
    simulator2.set_hrtf("tests/Test.sofa")
    simulator2.add_source(y3, -path)
    simulator2.add_source(y4, path)

    print('Simulating')

    audio1 = simulator1.simulate()
    
    t1 = time.time()
    audio2 = simulator2.simulate()
    t2 = time.time()

    print('time per speaker:', (t2 -t1)/2, 's')
    
    for s in range(audio1.shape[0]):
        write_audio_file(f'tests/1output{s}.wav', audio1[s], fs)

    for s in range(audio2.shape[0]):
        write_audio_file(f'tests/2output{s}.wav', audio2[s], fs)

def test_simulator():
    import librosa
    from scipy.io.wavfile import write as wavwrite
    import time
    import matplotlib.pyplot as plt

    def read_audio_file(file_path, sr):
        """
        Reads audio file to system memory.
        """
        return librosa.core.load(file_path, mono=False, sr=sr)[0]

    def read_with_pad(path, fs):
        y = read_audio_file(path, fs)
        if y.shape[-1] < T * fs:
            y = np.pad(y, (0, 0, T * fs - y.shape[-1]))
        else:
            y = y[:T * fs]

        return y

    def write_audio_file(file_path, data, sr):
        """
        Writes audio file to system memory.
        @param file_path: Path of the file to write to
        @param data: Audio signal to write (n_channels x n_samples)
        @param sr: Sampling rate
        """
        wavwrite(file_path, sr, data.T)

    fs = 8000
    T = 5
    fs = 8000
    sim = CIPICMotionSimulator2('data/360-BRIR-FOAIR-database/Binaural/SOFA/val.txt', fs)
    y1 = read_with_pad('tests/speech_samples/speech1.wav', fs)
    y2 = read_with_pad('tests/speech_samples/speech2.wav', fs)
    y3 = read_with_pad('tests/speech_samples/speech3.wav', fs)

    src, noise, path_src, noise_path = sim.simulate(np.array([y1, y2]), np.array([y3]), seed=3, debug=True)
    src = np.concatenate([src, np.expand_dims(noise, axis=0)], axis=0)
    print(src.shape)

    for i in range(src.shape[0]):
        write_audio_file(f'tests/output_src{i}.wav', src[i], fs)

    t = np.linspace(0, 1, len(path_src[0][:, 0]))
    plt.scatter(path_src[0][:, 0], path_src[0][:, 1], c=t)
    plt.savefig("tests/Source1_path.png")
    plt.clf()

    plt.scatter(path_src[1][:, 0], path_src[1][:, 1], c=t)
    plt.savefig("tests/Source2_path.png")
    plt.clf()

    plt.scatter(noise_path[:, 0], noise_path[:, 1], c=t)
    plt.savefig("tests/Source3_path.png")
    plt.clf()

def test_front_facing():
    import librosa
    from scipy.io.wavfile import write as wavwrite
    import time

    def read_audio_file(file_path, sr):
        """
        Reads audio file to system memory.
        """
        return librosa.core.load(file_path, mono=False, sr=sr)[0]

    def read_with_pad(path, fs):
        y = read_audio_file(path, fs)
        if y.shape[-1] < T * fs:
            y = np.pad(y, (0, 0, T * fs - y.shape[-1]))
        else:
            y = y[:T * fs]

        return y

    def write_audio_file(file_path, data, sr):
        """
        Writes audio file to system memory.
        @param file_path: Path of the file to write to
        @param data: Audio signal to write (n_channels x n_samples)
        @param sr: Sampling rate
        """
        wavwrite(file_path, sr, data.T)

    T = 5
    dt = 0.1
    fs = 8000

    t = np.arange(0, T, dt)
    rho = np.ones_like(t)
    phi = np.ones_like(t) * np.pi / 2

    theta = np.zeros_like(t)
    
    x = np.sin(phi) * np.cos(theta) * rho
    y = np.sin(phi) * np.sin(theta) * rho
    z = np.cos(phi) * rho

    plt.scatter(t, x, label='x', marker='x')
    plt.scatter(t, y, label='y', marker='o')
    plt.scatter(t, z, label='z', marker='.')
    plt.grid()
    plt.legend()
    plt.show()

    path = np.array([x, y, z]).T

    y1 = read_with_pad('tests/speech_samples/speech1.wav', fs)

    simulator1 = MotionSimulator(fs, dt)
    print('INITIALIZED')
    simulator1.set_hrtf("tests/Test.sofa")
    print('ADDSOURCE')
    simulator1.add_source(y1, path)

    print('Simulating')

    audio1 = simulator1.simulate()
    
    for s in range(audio1.shape[0]):
        write_audio_file(f'tests/ff_output{s}.wav', audio1[s], fs)

def test_path_generation():
    dt = 0.025
    simulator = CIPICMotionSimulator2('data/CIPIC_HRTF/hrtf_train.txt', 16000, dt)
    
    rng = random.Random(2)
    t = np.arange(0, 5, dt)
    path, theta, phi, prob, max_prob = simulator.get_piecewise_arc_path(rng, t, debug=True)

    plt.plot(t, np.rad2deg(theta), label='Theta')
    plt.plot(t, np.rad2deg(phi), label='phi')
    plt.plot(t, max_prob * 100, label='max_prob')
    plt.plot(t, prob * 100, label='probability')
    plt.legend()

    fig, ax = plt.subplots()
    ax = plt.figure().add_subplot(projection='3d')
    x, y, z = path.T
    ax.scatter(x, y, z, c=t)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    
    import librosa
    from scipy.io.wavfile import write as wavwrite

    def read_audio_file(file_path, sr):
        """
        Reads audio file to system memory.
        """
        return librosa.core.load(file_path, mono=False, sr=sr)[0]

    def read_with_pad(path, fs):
        y = read_audio_file(path, fs)
        if y.shape[-1] < T * fs:
            y = np.pad(y, (0, 0, T * fs - y.shape[-1]))
        else:
            y = y[:T * fs]

        return y

    def write_audio_file(file_path, data, sr):
        """
        Writes audio file to system memory.
        @param file_path: Path of the file to write to
        @param data: Audio signal to write (n_channels x n_samples)
        @param sr: Sampling rate
        """
        wavwrite(file_path, sr, data.T)

    fs = 16000
    T = 5
    y1 = read_with_pad('tests/speech_samples/speech1.wav', fs)
    
    simulator.set_hrtf("tests/Test.sofa")
    simulator._simulator.add_source(y1, path)
    out = simulator._simulator.simulate()[0]
    
    write_audio_file('tests/out.wav', out, fs)
    
    plt.show()


if __name__ == "__main__":
    # test_bindings()
    # test_simulator()
    # test_front_facing()
    test_path_generation()