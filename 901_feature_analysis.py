import os
import time
import datetime
import logging
import numpy as np
from umap import UMAP
import scipy.signal as sig
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pickle

from utils.data_handling import LaserDataInteractorESRF
#from utils.data_preparation import DataPrepper901
import utils.data_processing as processor
import matplotlib
from datetime import datetime
matplotlib.use('TkAgg')  # Use a GUI backend for plotting

#==============================================================================
# CONFIGURATION MANAGEMENT
#==============================================================================

class Config:
    """
    Config class for ESRF 901.
    This class encapsulates all parameters required for data processing, analysis,
    and visualization. Adjust any of the parameters below to tailor the analysis.
    """
    # Experiment number and Data File Configuration
    PATH_DATA_FOLDER = '/Volumes/Sandisk_SD/Work/IZFP/Laser/03_ESRF/Code/measurements/ESRF_TU_Ilmenau_IZFP/Messdaten_QASS'
    EXP_NO = 901
    FILE_NO = 18

    # Experiment Parameters
    SAMPLE_RATE = 6.25*10**6       # Sampling rate of the data [Hz] 
    WELD_START = 0.02              # Start time of the pulse welding [s] (determined "by ey")
    PULSE_DURATION = 11.25*10**-3  # Duration of the pulse welding [s]
    PULSE_LENGTH = int(PULSE_DURATION*SAMPLE_RATE)  # Length of the pulse welding [samples]
    
    # Logging Configuration
    LOG_DIR = "log"
    date = datetime.now().strftime("%Y-%m-%d")
    LOG_FILE = os.path.join(LOG_DIR, f"feature_analysis_{EXP_NO}_{date}.log")

    # Spectrogram Parameters
    SPECTROGRAM_WINDOW_SIZE_MULTIPLIER = 2**15  # Window size for FFT
    NFFT = SPECTROGRAM_WINDOW_SIZE_MULTIPLIER       # Number of FFT points

    # RMS Analysis Parameters
    RMS_WINDOW_Duration = 10*10**-3  # Window duration for RMS calculation [s]
    RMS_WINDOW_SIZE = int(RMS_WINDOW_Duration*SAMPLE_RATE)#10000  # Window size for sliding RMS calculation

    # Downsampling and Segmentation Parameters
    DOWNSAMPLE_FACTOR = 4
    SEGMENT_LENGTH_MS = 1                   # Segment length in milliseconds
    SEGMENT_OVERLAP_PERCENTAGE = 0.9          # 90% overlap between segments

    # Dimensionality Reduction Parameters
    PCA_N_COMPONENTS = 2
    PCA_FEATURE_IMPORTANCE_N_COMPONENTS = 3
    PCA_FEATURE_IMPORTANCE_TOP_FEATURES = 10

    TSNE_N_COMPONENTS = 2
    TSNE_PERPLEXITY = 30
    TSNE_N_ITER = 1000
    TSNE_RANDOM_STATE = 42

    UMAP_N_COMPONENTS = 2
    UMAP_N_NEIGHBORS = 15
    UMAP_MIN_DIST = 0.1
    UMAP_RANDOM_STATE = 42

    # Frequency Domain Feature Extraction Parameters
    FREQUENCY_BANDS = [
        (0, 1000),           # 0-1 kHz (low frequency)
        (1000, 10000),       # 1-10 kHz (low-mid frequency)
        (10000, 50000),      # 10-50 kHz (mid frequency)
        (50000, 100000),     # 50-100 kHz (high-mid frequency)
        (100000, 500000),    # 100-500 kHz (high frequency) <- it seems up to 450 kHz is useful for us
        (500000, 5e6 / (2 * DOWNSAMPLE_FACTOR))  # 500 kHz - Nyquist (very high frequency)
    ]
    N_PEAKS = 5
    INDEX_ANNOTATION_INTERVAL = 50

    # Change Point Detection Parameters
    CP_PELT_PENALTY_RMS_VAR = 10   # Penalty for RMS and variance features
    CP_PELT_PENALTY_ENTROPY = 15   # Penalty for entropy feature
    CP_BINSEG_N_BKPS = 5
    CP_BOTTOMUP_N_BKPS = 4
    CP_CONSENSUS_FEATURE_TYPES = ['rms', 'var', 'peak', 'entropy']
    CP_CONSENSUS_ALGORITHMS = ['pelt', 'binseg']
    CP_CONSENSUS_MODELS = ['l2', 'rbf']
    CP_CONSENSUS_N_BKPS = 5
    CP_CONSENSUS_PENALTY = 10
    CP_CONSENSUS_MIN = 0.3  # Require 30% consensus among methods
    
    SAVE_RESULTS = True
    SAVE_DIR = "results"
    EXPERIMENT_NAME = "901_feature_analysis"

#==============================================================================
# LOGGING SETUP
#==============================================================================

# Ensure the log directory exists
if not os.path.exists(Config.LOG_DIR):
    os.makedirs(Config.LOG_DIR)

# Configure logger with both console and file handlers
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
file_handler = logging.FileHandler(Config.LOG_FILE)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.info("Starting Ultrasonic Welding Data Analysis")

#==============================================================================
# DATA LOADING AND EVENT EXTRACTION
#==============================================================================

try:
    logger.info(f"Loading data file: {Config.EXP_NO}: in {Config.PATH_DATA_FOLDER}")

    # Get raw data
    interactor = LaserDataInteractorESRF()
    interactor.path_rel = Config.PATH_DATA_FOLDER
    interactor.fname = (1, Config.FILE_NO) # ChNo, fileNo
    sig1 = interactor.load_data(ret_dt=False)
    sig1 -= sig1.mean()
    interactor.fname = (2, Config.FILE_NO) # ChNo, fileNo
    sig2 = interactor.load_data(ret_dt=False)
    sig2 -= sig2.mean()
    # Combine the two channels, such that the first row is sig1 and the second row is sig 2
    raw_data = np.vstack((sig1, sig2))
    logger.info("Data file loaded successfully. Raw data shape: %s", raw_data.shape)
    
    # # Extract the signal segment corresponding to the event range
    # signal = raw_data[start_index:stop_index]
    # logger.info("Extracted signal segment from index %d to %d", start_index, stop_index)
except Exception as e:
    logger.exception("Error during data loading and event extraction")
    raise

#==============================================================================
# TIME DOMAIN VISUALIZATION
#==============================================================================

try:
    logger.info("Plotting time domain signal...")
    t = np.arange(raw_data.shape[1]) / Config.SAMPLE_RATE
    plt.figure(figsize=(15, 5))
    plt.plot(t, raw_data[0, :], label='Ch.1')
    plt.plot(t, raw_data[1, :], label='Ch.2')
    plt.title('Raw Ultrasonic Welding Signal in Time Domain')
    plt.xlabel('Time (seconds)')
    plt.ylabel('Amplitude [V]')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.show(block=True)  # Ensure the plot is displayed and blocks execution until closed
    logger.info("Time domain plot completed.")
except Exception as e:
    logger.exception("Error during time domain visualization")


#==============================================================================
# FREQUENCY DOMAIN ANALYSIS (FFT)
#==============================================================================
try:
    logger.info("Starting FFT analysis...")
    start_fft = time.time()

    fft_result = np.fft.fft(raw_data, axis=1)
    frequency_axis = np.fft.fftfreq(raw_data.shape[1], 1/Config.SAMPLE_RATE)
    positive_freq_mask = frequency_axis > 0  # Only show positive frequencies
    fix, axs = plt.subplots(2, 1, figsize=(15, 10))
    axs[0].plot(frequency_axis[positive_freq_mask], 20*np.log10(np.abs(fft_result[0, positive_freq_mask])), label='Ch.1')
    axs[0].set_title('Frequency Spectrum of Welding Signal (Ch.1)')
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('Amplitude [dB]')
    axs[0].grid(True)       
    axs[1].plot(frequency_axis[positive_freq_mask], 20*np.log10(np.abs(fft_result[1, positive_freq_mask])), label='Ch.2')
    axs[1].set_title('Frequency Spectrum of Welding Signal (Ch.2)')
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_ylabel('Amplitude [dB]')
    axs[1].grid(True)
    plt.show()
    fft_time = time.time() - start_fft
    logger.info("FFT analysis completed in %.2f seconds.", fft_time)
except Exception as e:
    logger.exception("Error during FFT analysis")

#==============================================================================
# SPECTROGRAM ANALYSIS
#==============================================================================

try:
    logger.info(f"Starting spectrogram analysis...")
    start_spec = time.time()

    fig, axs = plt.subplots(2, 1, figsize=(15, 10))

    for _ch in range(2):
        frequencies, times_spec, spectrogram = sig.spectrogram(
            raw_data[_ch, :],
            Config.SAMPLE_RATE,
            nperseg=Config.SPECTROGRAM_WINDOW_SIZE_MULTIPLIER,
            nfft=Config.NFFT
        )
        # Define frequency range for visualization
        freq_start = 0
        freq_end = 1000e3
        # Extract frequency indices of interest
        start_freq_index = np.where(frequencies >= freq_start)[0][0]
        end_freq_index = np.where(frequencies <= freq_end)[0][-1]
        spectrogram_subset = spectrogram[start_freq_index:end_freq_index + 1, :]
        frequencies_subset = frequencies[start_freq_index:end_freq_index + 1]
        
        axs[_ch].pcolormesh(times_spec, frequencies_subset, 10 * np.log10(spectrogram_subset), shading='gouraud')
        axs[_ch].set_ylabel('Frequency [MHz]', fontsize=12)
        axs[_ch].set_xlabel('Time [sec]', fontsize=12)
        axs[_ch].set_ylim(freq_start, freq_end)
        cbar = plt.colorbar(axs[_ch].collections[0], ax=axs[_ch], orientation='vertical')
        cbar.set_label('Power/Frequency (dB/Hz)', fontsize=12)
        # axs[_ch].colorbar(label='Power/Frequency (dB/Hz)')
        axs[_ch].set_title(f'Ch{_ch+1}')

    plt.tight_layout()
    plt.show()
    
    spec_time = time.time() - start_spec
    logger.info("Spectrogram analysis completed in %.2f seconds.", spec_time)
except Exception as e:
    logger.exception("Error during spectrogram analysis")

#==============================================================================
# RMS ANALYSIS
#==============================================================================

try:
    logger.info(f"Starting RMS analysis with {Config.RMS_WINDOW_SIZE}={Config.RMS_WINDOW_Duration*10**3}ms")
    start_rms = time.time()
    
    # Calculate the RMS of the signal using a sliding window
    cumsum_squares = np.cumsum(raw_data**2, axis=1) # shape = (2, N)
    rms_values = np.sqrt((cumsum_squares[:, Config.RMS_WINDOW_SIZE:] - cumsum_squares[:, :-Config.RMS_WINDOW_SIZE]) / Config.RMS_WINDOW_SIZE)
    
    time_seconds = np.arange(rms_values.shape[1]) / Config.SAMPLE_RATE
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_seconds, rms_values[0, :], label='Ch.1')
    plt.plot(time_seconds, rms_values[1, :], label='Ch.2')
    plt.title(f'Root Mean Square (RMS) with {Config.RMS_WINDOW_SIZE}={Config.RMS_WINDOW_Duration*10**3}ms')
    plt.legend(loc='upper right')
    plt.xlabel('Time (seconds)')
    plt.ylabel('RMS Amplitude')
    plt.grid(True)
    plt.show()
    
    rms_time = time.time() - start_rms
    logger.info("RMS analysis completed in %.2f seconds.", rms_time)
except Exception as e:
    logger.exception("Error during RMS analysis")

#==============================================================================
# SIGNAL SEGMENTATION AND DIMENSIONALITY REDUCTION -> To-Do (v250321) Combine my code from vpen vs WST correlation
#==============================================================================
"""
try:
    logger.info("Starting signal segmentation and downsampling...")
    start_seg = time.time()
    
    # Downsample the signal using the configured downsample factor
    signal_downsampled = dpu.filter_and_downsample(
        signal,
        fs=Config.SAMPLE_RATE,
        order=5,
        downsample_factor=Config.DOWNSAMPLE_FACTOR
    )
    
    # Create segments using the configured segmentation parameters
    segments_array = dpu.segment_signal(
        signal_downsampled,
        original_sampling_rate=Config.SEGMENT_ORIGINAL_SAMPLING_RATE,
        segment_length_ms=Config.SEGMENT_LENGTH_MS,
        overlap_percentage=Config.SEGMENT_OVERLAP_PERCENTAGE,
        downsample_factor=Config.DOWNSAMPLE_FACTOR
    )
    
    # Compute segment groups based on the template fractions
    n_segments = segments_array.shape[0]
    segment_groups = {}
    for group, (start_frac, end_frac) in Config.SEGMENT_GROUPS_TEMPLATE.items():
        start_idx = int(n_segments * start_frac)
        end_idx = int(n_segments * end_frac)
        segment_groups[group] = list(range(start_idx, end_idx))
    
    seg_time = time.time() - start_seg
    logger.info("Signal segmentation and downsampling completed in %.2f seconds.", seg_time)
except Exception as e:
    logger.exception("Error during signal segmentation and dimensionality reduction")
    raise

#--------------------------------------
# PCA Analysis of Signal Segments
#--------------------------------------

try:
    logger.info("Starting PCA analysis on signal segments...")
    start_pca = time.time()
    
    pca = PCA(n_components=Config.PCA_N_COMPONENTS)
    pca_result = pca.fit_transform(segments_array)
    
    plt.figure(figsize=(12, 10))
    
    # Determine segments not in any defined group
    all_segments = set(range(pca_result.shape[0]))
    grouped_segments = set()
    for group in segment_groups.values():
        grouped_segments.update(group)
    other_segments = list(all_segments - grouped_segments)
    
    # Plot "other" segments in the background
    for i in other_segments:
        plt.scatter(pca_result[i, 0], pca_result[i, 1], color=Config.GROUP_COLORS["Other"], alpha=0.2, s=30)
    
    # Plot each segment group with specific colors and annotate indices
    for group_name, group_indices in segment_groups.items():
        group_data = pca_result[group_indices]
        plt.scatter(group_data[:, 0], group_data[:, 1],
                    color=Config.GROUP_COLORS[group_name],
                    label=group_name,
                    alpha=0.8,
                    s=50)
        for i, idx in enumerate(group_indices):
            if len(group_indices) > 0 and i % (max(1, len(group_indices) // 5)) == 0:
                plt.text(pca_result[idx, 0], pca_result[idx, 1], str(idx), fontsize=10)
    
    plt.legend(fontsize=12)
    plt.xlabel('Principal Component 1', fontsize=14)
    plt.ylabel('Principal Component 2', fontsize=14)
    plt.title('PCA of Segmented Welding Data with Highlighted Welding Phases', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    pca_time = time.time() - start_pca
    logger.info("PCA analysis completed in %.2f seconds.", pca_time)
except Exception as e:
    logger.exception("Error during PCA analysis")

#--------------------------------------
# t-SNE Analysis of Signal Segments
#--------------------------------------

try:
    logger.info("Starting t-SNE analysis on signal segments...")
    start_tsne = time.time()
    
    tsne = TSNE(
        n_components=Config.TSNE_N_COMPONENTS,
        perplexity=Config.TSNE_PERPLEXITY,
        n_iter=Config.TSNE_N_ITER,
        random_state=Config.TSNE_RANDOM_STATE
    )
    tsne_result = tsne.fit_transform(segments_array)
    
    plt.figure(figsize=(12, 10))
    
    # Plot "other" segments in the background
    for i in other_segments:
        plt.scatter(tsne_result[i, 0], tsne_result[i, 1], color=Config.GROUP_COLORS["Other"], alpha=0.2, s=30)
    
    # Plot defined groups with annotations
    for group_name, group_indices in segment_groups.items():
        group_data = tsne_result[group_indices]
        plt.scatter(group_data[:, 0], group_data[:, 1],
                    color=Config.GROUP_COLORS[group_name],
                    label=group_name,
                    alpha=0.8,
                    s=50)
        for i, idx in enumerate(group_indices):
            if len(group_indices) > 0 and i % (max(1, len(group_indices) // 5)) == 0:
                plt.text(tsne_result[idx, 0], tsne_result[idx, 1], str(idx), fontsize=10)
    
    plt.legend(fontsize=12)
    plt.xlabel('t-SNE Component 1', fontsize=14)
    plt.ylabel('t-SNE Component 2', fontsize=14)
    plt.title('t-SNE of Segmented Welding Data with Highlighted Welding Phases', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    tsne_time = time.time() - start_tsne
    logger.info("t-SNE analysis completed in %.2f seconds.", tsne_time)
except Exception as e:
    logger.exception("Error during t-SNE analysis")

#--------------------------------------
# UMAP Analysis of Signal Segments
#--------------------------------------

try:
    logger.info("Starting UMAP analysis on signal segments...")
    start_umap = time.time()
    
    umap_model = UMAP(
        n_components=Config.UMAP_N_COMPONENTS,
        n_neighbors=Config.UMAP_N_NEIGHBORS,
        min_dist=Config.UMAP_MIN_DIST,
        random_state=Config.UMAP_RANDOM_STATE
    )
    umap_result = umap_model.fit_transform(segments_array)
    
    plt.figure(figsize=(12, 10))
    
    # Plot "other" segments in the background
    for i in other_segments:
        plt.scatter(umap_result[i, 0], umap_result[i, 1], color=Config.GROUP_COLORS["Other"], alpha=0.2, s=30)
    
    # Plot defined groups with annotations
    for group_name, group_indices in segment_groups.items():
        group_data = umap_result[group_indices]
        plt.scatter(group_data[:, 0], group_data[:, 1],
                    color=Config.GROUP_COLORS[group_name],
                    label=group_name,
                    alpha=0.8,
                    s=50)
        for i, idx in enumerate(group_indices):
            if len(group_indices) > 0 and i % (max(1, len(group_indices) // 5)) == 0:
                plt.text(umap_result[idx, 0], umap_result[idx, 1], str(idx), fontsize=10)
    
    plt.legend(fontsize=12)
    plt.xlabel('UMAP Component 1', fontsize=14)
    plt.ylabel('UMAP Component 2', fontsize=14)
    plt.title('UMAP of Segmented Welding Data with Highlighted Welding Phases', fontsize=16)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()
    
    umap_time = time.time() - start_umap
    logger.info("UMAP analysis completed in %.2f seconds.", umap_time)
except Exception as e:
    logger.exception("Error during UMAP analysis")

#==============================================================================
# FREQUENCY DOMAIN FEATURE EXTRACTION AND ANALYSIS
#==============================================================================

try:
    logger.info("Starting frequency domain feature extraction and analysis...")
    start_freq_analysis = time.time()
    
    # Define Nyquist frequency based on downsampled sampling rate
    nyquist_frequency = Config.SAMPLE_RATE / (2 * Config.DOWNSAMPLE_FACTOR)
    freq_range = (0, nyquist_frequency)
    
    analysis_results = dau.frequency_domain_analysis(
        segments_array=segments_array,
        fs=Config.SAMPLE_RATE / Config.DOWNSAMPLE_FACTOR,
        freq_bands=Config.FREQUENCY_BANDS,
        segment_groups=segment_groups,
        freq_range=freq_range,
        n_peaks=Config.N_PEAKS,
        index_annotation_interval=Config.INDEX_ANNOTATION_INTERVAL
    )
    
    # Extract features and feature names from analysis results
    features = analysis_results['features']
    feature_names = analysis_results['feature_names']
    
    print("\nFrequency Domain Feature Statistics:")
    print("-" * 50)
    for i, feature_name in enumerate(feature_names):
        feature_values = features[:, i]
        print(f"{feature_name}:")
        print(f"  Mean: {np.mean(feature_values):.4f}")
        print(f"  Std Dev: {np.std(feature_values):.4f}")
        print(f"  Min: {np.min(feature_values):.4f}")
        print(f"  Max: {np.max(feature_values):.4f}")
        print()
    
    # Analyze and report dominant peak frequency statistics
    peak_frequencies = analysis_results['peak_frequencies']
    print("\nDominant Frequency Statistics:")
    print("-" * 50)
    for i in range(peak_frequencies.shape[1]):
        peak_values = peak_frequencies[:, i]
        print(f"Peak {i+1}:")
        print(f"  Mean: {np.mean(peak_values)/1000:.2f} kHz")
        print(f"  Std Dev: {np.std(peak_values)/1000:.2f} kHz")
        print(f"  Min: {np.min(peak_values)/1000:.2f} kHz")
        print(f"  Max: {np.max(peak_values)/1000:.2f} kHz")
        print()
    
    freq_analysis_time = time.time() - start_freq_analysis
    logger.info("Frequency domain analysis completed in %.2f seconds.", freq_analysis_time)
except Exception as e:
    logger.exception("Error during frequency domain feature extraction and analysis")

print("\nAvailable frequency features:")
for i, feature in enumerate(feature_names):
    print(f"{i}: {feature}")

#--------------------------------------
# PCA FEATURE IMPORTANCE ANALYSIS
#--------------------------------------

try:
    logger.info("Starting PCA feature importance analysis...")
    start_pca_importance = time.time()
    
    print("\n" + "-"*50)
    print("PCA FEATURE IMPORTANCE ANALYSIS")
    print("-"*50)
    
    # Run PCA feature importance analysis
    pca_feature_importance_results = dau.analyze_pca_feature_importance(
        features,
        feature_names=feature_names,
        segment_groups=segment_groups,
        n_components=Config.PCA_FEATURE_IMPORTANCE_N_COMPONENTS,
        top_features=Config.PCA_FEATURE_IMPORTANCE_TOP_FEATURES,
        figsize=(14, 12)
    )
    
    # Save the feature importance analysis results if needed
    if Config.SAVE_RESULTS:
        if not os.path.exists(Config.SAVE_DIR):
            os.makedirs(Config.SAVE_DIR)
        importance_results_path = os.path.join(Config.SAVE_DIR, f"{Config.EXPERIMENT_NAME}_pca_feature_importance.pkl")
        with open(importance_results_path, 'wb') as f:
            pickle.dump(pca_feature_importance_results, f)
        logger.info(f"PCA feature importance results saved to: {importance_results_path}")
    
    pca_importance_time = time.time() - start_pca_importance
    logger.info("PCA feature importance analysis completed in %.2f seconds.", pca_importance_time)
    
    print("\nPCA Feature Importance Analysis Complete")
    print("-"*50)
except Exception as e:
    logger.exception("Error during PCA feature importance analysis")

#==============================================================================
# FEATURE VISUALIZATION EXAMPLES
#==============================================================================

try:
    logger.info("Starting feature visualization examples...")
    # Visualize PCA of all frequency features
    print("\nVisualizing PCA of all frequency features...")
    dau.visualize_feature_regions(features, feature_names, segment_groups)
    
    # Visualize PCA of Spectral Kurtosis feature
    print("\nVisualizing PCA of Spectral Kurtosis feature...")
    dau.visualize_feature_regions(features, feature_names, segment_groups, "Spectral Kurtosis")
    
    # Visualize PCA of Dominant Frequency feature
    print("\nVisualizing PCA of Dominant Frequency feature...")
    dau.visualize_feature_regions(features, feature_names, segment_groups, "Dominant Frequency (Hz)")
    
    # Visualize PCA of Harmonic Ratio feature
    print("\nVisualizing PCA of Harmonic Ratio feature...")
    dau.visualize_feature_regions(features, feature_names, segment_groups, "Harmonic Ratio")
    
    # Visualize PCA of multiple related spectral features
    print("\nVisualizing PCA of related spectral features...")
    dau.visualize_feature_regions(features, feature_names, segment_groups,
                                  ["Spectral Centroid (Hz)", "Spectral Bandwidth (Hz)", "Spectral Flatness"])
    logger.info("Feature visualization completed.")
except Exception as e:
    logger.exception("Error during feature visualization examples")

#==============================================================================
# CHANGE POINT DETECTION ANALYSIS
#==============================================================================

try:
    logger.info("Starting change point detection analysis...")
    start_cp = time.time()
    
    # Initialize dictionary to store change point results
    change_point_results = {
        'pelt': {},      # Pruned Exact Linear Time algorithm
        'binseg': {},    # Binary Segmentation algorithm
        'bottomup': {}   # Bottom-up segmentation algorithm
    }
    
    #--------------------------------------
    # PELT Algorithm Analysis
    #--------------------------------------
    logger.info("Detecting change points using PELT algorithm...")
    change_point_results['pelt']['rms'] = cpu.detect_change_points_pelt(
        segments_array, feature_type='rms', model='rbf', penalty=Config.CP_PELT_PENALTY_RMS_VAR
    )
    change_point_results['pelt']['var'] = cpu.detect_change_points_pelt(
        segments_array, feature_type='var', model='rbf', penalty=Config.CP_PELT_PENALTY_RMS_VAR
    )
    change_point_results['pelt']['entropy'] = cpu.detect_change_points_pelt(
        segments_array, feature_type='entropy', model='rbf', penalty=Config.CP_PELT_PENALTY_ENTROPY
    )
    
    #--------------------------------------
    # Binary Segmentation Analysis
    #--------------------------------------
    logger.info("Detecting change points using Binary Segmentation algorithm...")
    change_point_results['binseg']['rms'] = cpu.detect_change_points_binseg(
        segments_array, feature_type='rms', model='l2', n_bkps=Config.CP_BINSEG_N_BKPS
    )
    change_point_results['binseg']['peak'] = cpu.detect_change_points_binseg(
        segments_array, feature_type='peak', model='l2', n_bkps=Config.CP_BINSEG_N_BKPS
    )
    
    #--------------------------------------
    # Bottom-up Segmentation Analysis
    #--------------------------------------
    logger.info("Detecting change points using Bottom-up segmentation algorithm...")
    change_point_results['bottomup']['multi'] = cpu.detect_change_points_bottomup(
        segments_array, feature_type='multi', model='l2', n_bkps=Config.CP_BOTTOMUP_N_BKPS
    )
    
    #--------------------------------------
    # Visualize Change Points for Each Method
    #--------------------------------------
    logger.info("Visualizing detected change points for each method...")
    for algorithm, features_cp in change_point_results.items():
        for feature_type, change_points in features_cp.items():
            if change_points:  # Only visualize if change points were detected
                plt.figure(figsize=(15, 6))
                plt.title(f'Change Points detected with {algorithm.upper()} using {feature_type.UPPER()}')
                
                # Extract and plot feature data (use RMS for multi-feature case)
                if feature_type == 'multi':
                    feature_data = cpu.extract_features(segments_array, 'rms')
                    plt.plot(feature_data, label='RMS')
                else:
                    feature_data = cpu.extract_features(segments_array, feature_type)
                    plt.plot(feature_data, label=feature_type.upper())
                
                # Plot vertical lines for each detected change point
                for cp in change_points:
                    plt.axvline(x=cp, color='r', linestyle='--')
                
                plt.xlabel('Segment Index')
                plt.ylabel('Feature Value')
                plt.grid(True)
                plt.legend()
                plt.show()
    
    #--------------------------------------
    # Comprehensive Change Point Analysis
    #--------------------------------------
    logger.info("Performing comprehensive change point analysis with consensus...")
    cp_analysis_results = cpu.change_point_analysis(
        segments_array,
        feature_types=Config.CP_CONSENSUS_FEATURE_TYPES,
        algorithms=Config.CP_CONSENSUS_ALGORITHMS,
        models=Config.CP_CONSENSUS_MODELS,
        n_bkps=Config.CP_CONSENSUS_N_BKPS,
        penalty=Config.CP_CONSENSUS_PENALTY,
        min_consensus=Config.CP_CONSENSUS_MIN
    )
    
    # Visualize the comprehensive change point analysis results
    cpu.visualize_change_point_analysis(segments_array, cp_analysis_results)
    
    consensus_points = cp_analysis_results['consensus_change_points']
    print(f"\nDetected {len(consensus_points)} consensus change points at indices: {consensus_points}")
    
    cp_time = time.time() - start_cp
    logger.info("Change point detection analysis completed in %.2f seconds.", cp_time)
except Exception as e:
    logger.exception("Error during change point detection analysis")

logger.info("Ultrasonic Welding Data Analysis COMPLETE")"
"""
