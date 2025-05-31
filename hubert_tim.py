import os
import sys
import json
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import wfdb
from scipy.signal import firwin, lfilter, resample_poly, butter, filtfilt
from scipy.io import loadmat
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset as TorchDataset, WeightedRandomSampler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.amp import autocast
from torch.cuda.amp import GradScaler
import torchmetrics
from tqdm import tqdm
from typing import List, Optional, Union, Tuple, Dict, Any
from transformers import HubertConfig as TransformersHubertConfig
from transformers.modeling_outputs import BaseModelOutput
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import multilabel_confusion_matrix, roc_auc_score, precision_recall_fscore_support, confusion_matrix, fbeta_score
import traceback
import warnings
warnings.filterwarnings('ignore')

# Hardware optimization imports
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import threading
import time
try:
    import psutil
    import GPUtil
    HARDWARE_MONITORING_AVAILABLE = True
except ImportError:
    HARDWARE_MONITORING_AVAILABLE = False
    print("Warning: psutil and/or GPUtil not installed. Hardware monitoring disabled.")
    print("Install with: pip install psutil gputil")

# ==============================================================================
# == 0. IMPROVED CONFIGURATION SETTINGS WITH HARDWARE OPTIMIZATIONS
# ==============================================================================

class Config:
    # Dataset information files
    MIMIC_IV_INFO_CSV = Path(r"D:\MIMIC IV Selected\MIMIC_IV_Selected_Information.csv")
    PHYSIONET_2021_INFO_CSV = Path(r"D:\Physionet 2021 Selected\Physionet_2021_Information.csv")
    ASSIGNMENT3_INFO_CSV = Path(r"D:\Assignment 3 Data\file_info_output.csv")
    ASSIGNMENT3_DATA_DIR = Path(r"D:\Assignment 3 Data")

    # Original HuBERT-ECG codebase (if needed)
    HUBERT_ECG_ORIGINAL_CODE_BASE_PATH = r"D:\ECG Models\HuBERT ECG"

    # Output directory
    OUTPUT_BASE_DIR = Path(r"D:\ECG Models\HuBERT ECG Result")
    PREPROCESS_FILES_DIR = Path(r"D:\ECG Models\HuBERT ECG Result\Preprocess Files")
    NPY_FILES_DIR = PREPROCESS_FILES_DIR
    EVALUATION_PLOTS_DIR = OUTPUT_BASE_DIR / "evaluation_plots"
    RESULTS_CSV_PATH = OUTPUT_BASE_DIR / "detailed_evaluation_results.csv"
    SAMPLE_PREDICTIONS_CSV_PATH = OUTPUT_BASE_DIR / "individual_sample_predictions.csv"
    CONFIDENCE_SCORES_CSV_PATH = OUTPUT_BASE_DIR / "confidence_scores.csv"
    ASSIGNMENT3_PREDICTIONS_CSV_PATH = OUTPUT_BASE_DIR / "assignment3_predictions.csv"
    CONFUSION_MATRICES_DIR = OUTPUT_BASE_DIR / "confusion_matrices"
    DATA_SPLIT_REPORT_PATH = OUTPUT_BASE_DIR / "data_split_report.csv"
    LOG_FILE_PATH = OUTPUT_BASE_DIR / "run_log.txt"

    # Condition mapping
    CONDITION_MAPPING = {
        "1st-degree AV block": "1st_degree_AV_block", "1st degree AV block": "1st_degree_AV_block", 
        "First-degree AV block": "1st_degree_AV_block", "First degree AV block": "1st_degree_AV_block",
        "1st Degree AV Block": "1st_degree_AV_block",
        "2nd-degree AV block": "2nd_degree_AV_block", "2nd degree AV block": "2nd_degree_AV_block", 
        "Second-degree AV block": "2nd_degree_AV_block", "Second degree AV block": "2nd_degree_AV_block",
        "2nd Degree AV Block": "2nd_degree_AV_block",
        "Atrial Fibrillation": "Atrial_Fibrillation", "atrial fibrillation": "Atrial_Fibrillation", "AF": "Atrial_Fibrillation",
        "Atrial Flutter": "Atrial_Flutter", "atrial flutter": "Atrial_Flutter", "AFL": "Atrial_Flutter",
        "Premature atrial contractions": "Premature_atrial_contractions", "premature atrial contractions": "Premature_atrial_contractions", "PAC": "Premature_atrial_contractions",
        "Premature ventricular contractions": "Premature_ventricular_contractions", "premature ventricular contractions": "Premature_ventricular_contractions", "PVC": "Premature_ventricular_contractions",
        "Sinus Bradycardia": "Sinus_Bradycardia", "sinus bradycardia": "Sinus_Bradycardia",
        "Sinus Tachycardia": "Sinus_Tachycardia", "sinus tachycardia": "Sinus_Tachycardia",
        "NORMAL": "NORMAL", "Normal": "NORMAL", "normal": "NORMAL",
        "Left bundle branch block": "Left_bundle_branch_block", "left bundle branch block": "Left_bundle_branch_block", "LBBB": "Left_bundle_branch_block",
        "Right bundle branch block": "Right_bundle_branch_block", "right bundle branch block": "Right_bundle_branch_block", "RBBB": "Right_bundle_branch_block"
    }

    TARGET_CONDITIONS = [
        "NORMAL", "Atrial_Fibrillation", "Atrial_Flutter",
        "Premature_atrial_contractions", "Premature_ventricular_contractions",
        "1st_degree_AV_block", "2nd_degree_AV_block",
        "Left_bundle_branch_block", "Right_bundle_branch_block"
    ]
    NUM_LABELS = len(TARGET_CONDITIONS)

    # Signal processing parameters
    TARGET_SR_PREPROCESS = 500
    TARGET_SR_MODEL = 100
    DOWNSAMPLING_FACTOR_MODEL = TARGET_SR_PREPROCESS // TARGET_SR_MODEL
    FILTER_BAND = [0.05, 47]
    SEGMENT_DURATION_SECONDS = 5
    NUM_LEADS = 12
    FINAL_INPUT_LENGTH = NUM_LEADS * TARGET_SR_MODEL * SEGMENT_DURATION_SECONDS

    SCALING_METHOD = "minmax"

    # Model configuration
    HUGGINGFACE_MODEL_NAME = "Edoardo-BS/hubert-ecg-large"

    # Data splitting - 40/40/20 (4/4/2)
    PATIENT_ID_COLUMN = 'patient_id'
    TRAIN_RATIO = 0.4
    VAL_RATIO = 0.4
    TEST_RATIO = 0.2
    RANDOM_STATE = 42

    # Augmentation parameters
    AUGMENTATION_ENABLED = True
    AUG_RANDOM_OFFSET_MAX_MS = 200
    AUG_GAUSSIAN_NOISE_STD = 0.01
    AUG_BASELINE_WANDER_MAX_AMPLITUDE = 0.05
    AUG_BASELINE_WANDER_FREQ_RANGE = (0.05, 0.5)
    AUG_LEAD_DROP_PROB = 0.1
    AUG_LEAD_MIXUP_PROB = 0.1

    # Training parameters - OPTIMIZED FOR RTX 5090 + 128GB RAM
    BATCH_SIZE = 512  # Increased from 64 for 128GB RAM
    GRADIENT_ACCUMULATION_STEPS = 1  # Keep at 1 since we can fit large batches
    PHYSICAL_BATCH_SIZE = BATCH_SIZE // GRADIENT_ACCUMULATION_STEPS
    USE_AMP = True
    
    # RTX 5090 optimization
    USE_BF16 = True  # RTX 5090 has excellent BF16 support
    AMP_DTYPE = torch.bfloat16 if USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    
    CLIP_GRAD_NORM = 1.0

    # Loss function
    LOSS_TYPE = "bce_with_pos_weight"
    FOCAL_LOSS_ALPHA = 0.25
    FOCAL_LOSS_GAMMA = 2.0

    # Two-stage fine-tuning
    NUM_EPOCHS_STAGE1 = 5
    LEARNING_RATE_STAGE1 = 1e-3
    NUM_EPOCHS_STAGE2 = 10
    LEARNING_RATE_STAGE2 = 1e-4
    UNFREEZE_LAYERS_COUNT = 3

    # LR Scheduler
    LR_SCHEDULER_WARMUP_EPOCHS = 1

    # Threshold Optimization
    FBETA_BETA = 2.0

    # Hardware optimization settings - OPTIMIZED FOR 128GB RAM
    NUM_WORKERS = 32  # Increased for 128GB RAM (was 8)
    PARALLEL_PREPROCESSING_CORES = multiprocessing.cpu_count()  # Use ALL 24 cores
    ENABLE_CUDNN_BENCHMARK = True  # Better GPU kernel selection
    ENABLE_TF32 = True  # RTX 5090 TF32 support
    CACHE_DATASET_IN_MEMORY = True  # Load entire dataset into RAM
    HARDWARE_MONITORING_INTERVAL = 30  # seconds
    
    # Additional 128GB RAM optimizations
    PREFETCH_FACTOR = 10  # Increased from default 2
    DATALOADER_MEMORY_PINNING = True
    PREPROCESSING_RAM_CACHE_GB = 50  # Use 50GB for preprocessing cache
    
    # Misc
    SKIP_PREPROCESSING_IF_NPY_EXISTS = True
    
    # Paths for trained components
    TRAINED_MODEL_PATH = OUTPUT_BASE_DIR / "trained_full_model.pth"
    OPTIMIZED_THRESHOLDS_PATH = OUTPUT_BASE_DIR / "optimized_thresholds.json"

config = Config()

# Ensure output directories exist
config.OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
config.PREPROCESS_FILES_DIR.mkdir(parents=True, exist_ok=True)
config.NPY_FILES_DIR.mkdir(parents=True, exist_ok=True)
config.EVALUATION_PLOTS_DIR.mkdir(parents=True, exist_ok=True)
config.CONFUSION_MATRICES_DIR.mkdir(parents=True, exist_ok=True)

# Setup basic logging to a file
class Logger(object):
    def __init__(self, filename=config.LOG_FILE_PATH):
        self.terminal = sys.stdout
        self.log_file_handle = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log_file_handle.write(message)

    def flush(self):
        self.terminal.flush()
        self.log_file_handle.flush()

    def isatty(self):
        return self.terminal.isatty()

sys.stdout = Logger(config.LOG_FILE_PATH)
sys.stderr = Logger(config.LOG_FILE_PATH)

print(f"--- Script started/restarted at {pd.Timestamp.now()} ---")
print(f"Hardware Optimization Settings:")
print(f"  - CPU Workers: {config.NUM_WORKERS}")
print(f"  - Batch Size: {config.BATCH_SIZE}")
print(f"  - AMP Dtype: {config.AMP_DTYPE}")
print(f"  - Parallel Preprocessing Cores: {config.PARALLEL_PREPROCESSING_CORES}")

# ==============================================================================
# == 1. HARDWARE MONITORING
# ==============================================================================
class HardwareMonitor:
    """Monitor CPU and GPU utilization during training"""
    
    def __init__(self, log_interval=30):
        self.log_interval = log_interval
        self.monitoring = False
        self.monitor_thread = None
        self.enabled = HARDWARE_MONITORING_AVAILABLE
        
    def start(self):
        """Start monitoring in background thread"""
        if not self.enabled:
            return
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()
        
    def stop(self):
        """Stop monitoring"""
        if not self.enabled:
            return
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
            
    def _monitor_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                # CPU metrics
                cpu_percent = psutil.cpu_percent(interval=1, percpu=True)
                cpu_avg = sum(cpu_percent) / len(cpu_percent)
                memory = psutil.virtual_memory()
                
                # GPU metrics
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]  # RTX 5090
                    gpu_util = gpu.load * 100
                    gpu_memory = gpu.memoryUsed / gpu.memoryTotal * 100
                    gpu_temp = gpu.temperature
                    
                    print(f"\n=== Hardware Utilization ===")
                    print(f"CPU: {cpu_avg:.1f}% avg (cores 0-7: {[f'{c:.0f}%' for c in cpu_percent[:8]]})")
                    print(f"RAM: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB / {memory.total/1024**3:.1f}GB) - Available: {memory.available/1024**3:.1f}GB")
                    print(f"GPU: {gpu_util:.1f}% | VRAM: {gpu_memory:.1f}% ({gpu.memoryUsed:.0f}MB / {gpu.memoryTotal:.0f}MB) | Temp: {gpu_temp}°C")
                    print("=" * 30)
                    
                    # Warnings
                    if gpu_util < 80:
                        print("⚠️  GPU utilization below 80% - consider increasing batch size")
                    if cpu_avg < 50:
                        print("⚠️  CPU utilization below 50% - consider increasing NUM_WORKERS")
                    if memory.percent < 30:  # Less than 30% of 128GB = ~38GB
                        print("⚠️  RAM utilization below 30% - consider enabling more caching or larger batches")
                        print(f"   You have {memory.available/1024**3:.1f}GB unused RAM!")
                
                time.sleep(self.log_interval)
            except Exception as e:
                print(f"Hardware monitoring error: {e}")
                time.sleep(self.log_interval)

# ==============================================================================
# == 2. MODEL CLASSES (HuBERTECGConfig, HuBERTECG, etc.)
# ==============================================================================
original_hubert_code_path = Path(config.HUBERT_ECG_ORIGINAL_CODE_BASE_PATH) / "code"
if original_hubert_code_path.exists() and str(original_hubert_code_path) not in sys.path:
    sys.path.insert(0, str(original_hubert_code_path))
    print(f"Added {original_hubert_code_path} to sys.path")

class HuBERTECGConfig(TransformersHubertConfig):
    model_type = "hubert_ecg"
    def __init__(self, ensemble_length: int = 1, vocab_sizes: List[int] = [100], **kwargs):
        kwargs.pop('trust_remote_code', None)
        super().__init__(**kwargs)
        self.ensemble_length = ensemble_length
        self.vocab_sizes = vocab_sizes if isinstance(vocab_sizes, list) else [vocab_sizes]

class HuBERTECG(nn.Module):
    def __init__(self, config_obj: HuBERTECGConfig):
        super().__init__()
        self.config = config_obj
        from transformers import HubertModel as TransformersHubertModel
        hf_compat_dict = {k: v for k, v in config_obj.to_dict().items()
                          if k not in ['ensemble_length', 'vocab_sizes', 'model_type', '_name_or_path']}
        essential_params = ['hidden_size', 'num_hidden_layers', 'num_attention_heads', 'intermediate_size']
        for p in essential_params:
            if p not in hf_compat_dict and hasattr(config_obj,p):
                 hf_compat_dict[p] = getattr(config_obj,p)
        hf_compat_config = TransformersHubertConfig(**hf_compat_dict)
        self.hubert_model = TransformersHubertModel(hf_compat_config)

    @classmethod
    def from_pretrained(cls, model_name_or_path, config_obj=None, ignore_mismatched_sizes=False, **kwargs):
        trust_remote_code_val = kwargs.pop('trust_remote_code', True)
        if config_obj is None:
            config_obj = HuBERTECGConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code_val, **kwargs)
        model = cls(config_obj)
        from transformers import HubertModel as TransformersHubertModel
        try:
            pretrained_hf_config = TransformersHubertConfig.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code_val)
            model.hubert_model = TransformersHubertModel.from_pretrained(
                model_name_or_path, config=pretrained_hf_config,
                ignore_mismatched_sizes=ignore_mismatched_sizes, trust_remote_code=trust_remote_code_val, **kwargs)
            print(f"Successfully loaded weights for internal HubertModel from {model_name_or_path}")
        except TypeError as te:
            if "got multiple values for keyword argument 'trust_remote_code'" in str(te):
                print("TypeError: 'trust_remote_code' passed multiple times. Retrying...")
                kwargs_for_hubert_model = {k:v for k,v in kwargs.items() if k != 'trust_remote_code'}
                model.hubert_model = TransformersHubertModel.from_pretrained(
                    model_name_or_path, config=pretrained_hf_config, ignore_mismatched_sizes=ignore_mismatched_sizes,
                    trust_remote_code=trust_remote_code_val, **kwargs_for_hubert_model)
                print(f"Successfully loaded weights (retry) for internal HubertModel from {model_name_or_path}")
            else: raise
        except Exception as e: 
            print(f"Could not load pretrained weights: {e}")
            raise
        return model

    def forward(self, input_values: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None,
                mask_time_indices: Optional[torch.FloatTensor] = None, output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None, return_dict: Optional[bool] = None) -> Union[Tuple, BaseModelOutput]:
        return self.hubert_model(input_values=input_values, attention_mask=attention_mask, output_attentions=output_attentions,
                                 output_hidden_states=output_hidden_states, return_dict=return_dict)

class ActivationFunction(nn.Module):
    def __init__(self, activation : str):
        super(ActivationFunction, self).__init__()
        self.activation = activation
        if activation == 'tanh': self.act = nn.Tanh()
        elif activation == 'relu': self.act = nn.ReLU()
        elif activation == 'gelu': self.act = nn.GELU()
        elif activation == 'sigmoid': self.act = nn.Sigmoid()
        else: raise ValueError('Activation function not supported')
    def forward(self, x): return self.act(x)

class HuBERTForECGClassification(nn.Module):
    def __init__(
        self, hubert_ecg_model: HuBERTECG, num_labels : int, classifier_hidden_size : int = None,
        activation : str = 'tanh', classifier_dropout_prob : float = 0.1):
        super(HuBERTForECGClassification, self).__init__()
        self.hubert_ecg = hubert_ecg_model
        self.num_labels = num_labels
        self.base_model_config = self.hubert_ecg.hubert_model.config
        if hasattr(self.base_model_config, 'mask_time_prob'): self.base_model_config.mask_time_prob = 0.0
        if hasattr(self.base_model_config, 'mask_feature_prob'): self.base_model_config.mask_feature_prob = 0.0
        self.classifier_dropout = nn.Dropout(classifier_dropout_prob)
        base_model_hidden_size = self.base_model_config.hidden_size
        if classifier_hidden_size is None:
            self.classifier = nn.Linear(base_model_hidden_size, num_labels)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(base_model_hidden_size, classifier_hidden_size),
                ActivationFunction(activation), nn.Dropout(classifier_dropout_prob),
                nn.Linear(classifier_hidden_size, num_labels))
    def forward(self, x: Optional[torch.Tensor], attention_mask: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None, output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None) -> torch.Tensor:
        encodings = self.hubert_ecg(input_values=x, attention_mask=attention_mask, output_attentions=output_attentions,
                                    output_hidden_states=output_hidden_states, return_dict=True)
        sequence_output = encodings.last_hidden_state
        pooled_output = sequence_output.mean(dim=1)
        pooled_output = self.classifier_dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

# ==============================================================================
# == 3. DATASET AND AUGMENTATION
# ==============================================================================
class ECGAugmentations:
    def __init__(self, target_sr=config.TARGET_SR_MODEL): 
        self.target_sr = target_sr
        
    def random_offset(self, d: np.ndarray) -> np.ndarray:
        m = int(config.AUG_RANDOM_OFFSET_MAX_MS/1000*self.target_sr)
        if m==0 or d.shape[1]<=m: return d
        o=np.random.randint(-m,m+1); a=np.zeros_like(d)
        if o>0: a[:,:-o]=d[:,o:]
        elif o<0: a[:,-o:]=d[:,:o]
        else: return d
        return a
        
    def gaussian_noise(self, d: np.ndarray) -> np.ndarray:
        if config.AUG_GAUSSIAN_NOISE_STD==0: return d
        s=config.AUG_GAUSSIAN_NOISE_STD*np.std(d[d!=0]) if np.any(d!=0) else config.AUG_GAUSSIAN_NOISE_STD
        return d+np.random.normal(0,s,d.shape) if s>0 else d
        
    def baseline_wander(self, d: np.ndarray) -> np.ndarray:
        if config.AUG_BASELINE_WANDER_MAX_AMPLITUDE==0: return d
        _,n=d.shape; t=np.linspace(0,n/self.target_sr,n,endpoint=False)
        for i in range(d.shape[0]):
            if np.all(d[i,:]==0): continue
            f=np.random.uniform(*config.AUG_BASELINE_WANDER_FREQ_RANGE)
            p=np.ptp(d[i,:]) if np.ptp(d[i,:])>1e-6 else 1.0
            a=config.AUG_BASELINE_WANDER_MAX_AMPLITUDE*p*np.random.uniform(0.5,1.0)
            ph=np.random.uniform(0,2*np.pi)
            d[i,:]+=a*np.sin(2*np.pi*f*t+ph)
        return d
        
    def lead_drop(self, d: np.ndarray) -> np.ndarray:
        if np.random.rand()<config.AUG_LEAD_DROP_PROB: 
            d[np.random.randint(0,d.shape[0]),:]=0
        return d
        
    def lead_mixup(self, d: np.ndarray) -> np.ndarray:
        if np.random.rand()<config.AUG_LEAD_MIXUP_PROB and d.shape[0]>=2:
            act_idx=[i for i,ld in enumerate(d) if not np.all(ld==0)]
            if len(act_idx)<2: return d
            l1,l2=np.random.choice(act_idx,2,replace=False); lam=np.random.beta(0.2,0.2)
            d[l1,:]=lam*d[l1,:]+(1-lam)*d[l2,:]
        return d
        
    def apply(self, d: np.ndarray) -> np.ndarray:
        a=d.copy(); a=self.random_offset(a); a=self.lead_drop(a); a=self.lead_mixup(a)
        a=self.gaussian_noise(a); a=self.baseline_wander(a); return a

class ECGDataset(TorchDataset):
    def __init__(self, dataframe, labels_column_names, npy_base_dir, is_train=False, augmentations: Optional[ECGAugmentations]=None):
        self.df,self.labels_cols,self.npy_dir=dataframe,labels_column_names,Path(npy_base_dir)
        self.is_train,self.aug=is_train,augmentations if config.AUGMENTATION_ENABLED and is_train else None
        
    def __len__(self): 
        return len(self.df)
        
    def __getitem__(self, idx):
        rec=self.df.iloc[idx]; npy_fn=rec['npy_path']; ecg_p=self.npy_dir/npy_fn
        cond_s,ds_s=rec.get('condition','Unknown'),rec.get('dataset','Unknown')
        try:
            ecg500=np.load(ecg_p).astype(np.float32)
        except FileNotFoundError:
            print(f"ERR: NPY not found: {ecg_p}",file=sys.stderr)
            return {'ecg_data':torch.zeros(config.FINAL_INPUT_LENGTH,dtype=torch.float32),
                    'labels':torch.zeros(len(self.labels_cols),dtype=torch.float32),
                    'condition_str':cond_s,'dataset_str':ds_s,'npy_path':npy_fn,'index':idx}
        nl,ns500=ecg500.shape; tgs100pl=config.TARGET_SR_MODEL*config.SEGMENT_DURATION_SECONDS; ecg100l=[]
        for i in range(nl):
            l500=np.ascontiguousarray(ecg500[i,:])
            rl=resample_poly(l500,1,config.DOWNSAMPLING_FACTOR_MODEL,window=('kaiser',5.0))
            if len(rl)>tgs100pl: rl=rl[:tgs100pl]
            elif len(rl)<tgs100pl: rl=np.concatenate([rl,np.zeros(tgs100pl-len(rl),dtype=np.float32)])
            ecg100l.append(rl)
        ecg100=np.array(ecg100l,dtype=np.float32)
        if self.aug: ecg100=self.aug.apply(ecg100)
        ecg_flat=ecg100.flatten(); lbl_v=rec[self.labels_cols].values.astype(np.float32)
        return {'ecg_data':torch.from_numpy(ecg_flat.copy()),'labels':torch.tensor(lbl_v,dtype=torch.float32),
                'condition_str':cond_s,'dataset_str':ds_s,'npy_path':npy_fn,'index':idx}

class CachedECGDataset(ECGDataset):
    """Dataset with in-memory caching for faster loading"""
    
    def __init__(self, *args, cache_in_memory=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.cache_in_memory = cache_in_memory if cache_in_memory is not None else config.CACHE_DATASET_IN_MEMORY
        self.cache = {}
        
        if self.cache_in_memory:
            print(f"Pre-loading {len(self)} samples into memory...")
            for idx in tqdm(range(len(self)), desc="Caching dataset"):
                self.cache[idx] = super().__getitem__(idx)
            print(f"Cached {len(self.cache)} samples in memory")
    
    def __getitem__(self, idx):
        if self.cache_in_memory and idx in self.cache:
            return self.cache[idx]
        return super().__getitem__(idx)

# ==============================================================================
# == 4. PREPROCESSING FUNCTIONS
# ==============================================================================
def preprocess_ecg_segment_for_hubert(sig_raw, orig_fs, fn_log=""):
    if sig_raw is None or sig_raw.size==0: 
        return np.zeros((config.NUM_LEADS,config.TARGET_SR_PREPROCESS*config.SEGMENT_DURATION_SECONDS),dtype=np.float32)
    sig=sig_raw.T if sig_raw.shape[0]>sig_raw.shape[1] else sig_raw.copy()
    cl=sig.shape[0]
    if cl!=config.NUM_LEADS:
        if cl>config.NUM_LEADS: sig=sig[:config.NUM_LEADS,:]
        elif 0<cl<config.NUM_LEADS: sig=np.concatenate([sig,np.zeros((config.NUM_LEADS-cl,sig.shape[1]),dtype=sig.dtype)],axis=0)
        elif cl==0: return np.zeros((config.NUM_LEADS,config.TARGET_SR_PREPROCESS*config.SEGMENT_DURATION_SECONDS),dtype=np.float32)
        else: raise ValueError(f"Invalid lead count {cl} for {fn_log}.")
    if orig_fs!=config.TARGET_SR_PREPROCESS:
        nso,nst=sig.shape[1],int(sig.shape[1]*(config.TARGET_SR_PREPROCESS/orig_fs))
        if nst==0 and nso>0: nst=1
        rsl=[]
        for i in range(sig.shape[0]):
            if nso>0:
                cd=sig[i,:].squeeze(); up,dn=config.TARGET_SR_PREPROCESS,orig_fs; com=np.gcd(up,dn)
                up//=com; dn//=com; rsl.append(resample_poly(cd,up,dn))
            else: rsl.append(np.zeros(nst,dtype=sig.dtype))
        ml=nst
        for i in range(len(rsl)):
            l=rsl[i]
            if len(l)>ml: rsl[i]=l[:ml]
            elif len(l)<ml: rsl[i]=np.pad(l,(0,ml-len(l)),mode='constant',constant_values=0)
        sig=np.array(rsl,dtype=sig.dtype) if rsl else np.zeros((config.NUM_LEADS,nst),dtype=sig.dtype)
    lc,hc=config.FILTER_BAND; nyq=0.5*config.TARGET_SR_PREPROCESS; l,h=lc/nyq,hc/nyq
    l,h=max(l,1e-6),min(h,1.0-1e-6); b,a=None,None
    if l<h:
        try: b,a=butter(3,[l,h],btype='band',analog=False)
        except ValueError: pass
    fsl=[]
    for i in range(sig.shape[0]):
        fsl.append(filtfilt(b,a,sig[i,:].squeeze()) if b is not None and a is not None and sig.shape[1]>10 else sig[i,:])
    fsig=np.array(fsl,dtype=sig.dtype) if fsl else sig.copy()
    sps=int(config.SEGMENT_DURATION_SECONDS*config.TARGET_SR_PREPROCESS)
    if fsig.shape[1]==0: return np.zeros((config.NUM_LEADS,sps),dtype=np.float32)
    seg=fsig[:,:sps] if fsig.shape[1]>=sps else np.concatenate([fsig,np.zeros((fsig.shape[0],sps-fsig.shape[1]),dtype=fsig.dtype)],axis=1)
    sc_seg=np.zeros_like(seg,dtype=np.float32)
    for i in range(seg.shape[0]):
        ld=seg[i,:]
        if config.SCALING_METHOD=="minmax":
            miv,mav=np.min(ld),np.max(ld)
            sc_seg[i,:]=2*(ld-miv)/(mav-miv)-1 if mav-miv>1e-8 else np.zeros_like(ld)
        elif config.SCALING_METHOD=="zscore":
            mev,sdv=np.mean(ld),np.std(ld)
            sc_seg[i,:]=(ld-mev)/sdv if sdv>1e-8 else np.zeros_like(ld)
        else: raise ValueError(f"Unknown scaling: {config.SCALING_METHOD}")
    return sc_seg

def load_ecg_data(fp_str: str, ext: str) -> Tuple[Optional[np.ndarray], Optional[int]]:
    fp=Path(fp_str)
    if ext=='.dat':
        try: 
            rec=wfdb.rdrecord(str(fp.with_suffix(''))); 
            return rec.p_signal.astype(np.float32),rec.fs
        except Exception: 
            return None,None
    elif ext=='.mat':
        try:
            mat=loadmat(fp); sig_mat=None
            for k in ['val','data']:
                if k in mat: sig_mat=mat[k]; break
            if sig_mat is None:
                nk=[k for k,v in mat.items() if not k.startswith('__') and isinstance(v,np.ndarray) and v.ndim>=2]
                if nk: sig_mat=mat[max(nk,key=lambda k:mat[k].size)]
                else: return None,None
            sig_mat=sig_mat.astype(np.float32)
            sig=sig_mat if (sig_mat.shape[0]>sig_mat.shape[1]*2 and sig_mat.shape[1]<=16) else sig_mat.T
            hea_p,fs=fp.with_suffix('.hea'),500
            if hea_p.exists():
                try:
                    with open(hea_p,'r') as f:
                        pts=f.readline().strip().split()
                        if len(pts)>2 and pts[2].isdigit(): fs=int(pts[2])
                        elif len(pts)>1 and pts[1].isdigit(): fs=int(pts[1])
                except Exception: pass
            return sig,fs
        except Exception: 
            return None,None
    elif ext=='.npy':
        try:
            data = np.load(fp).astype(np.float32)
            if data.shape[0] != 12 and data.shape[1] == 12:
                data = data.T
            return data, 100
        except Exception:
            return None, None
    return None,None

def extract_patient_id(fp_str: str, fn: str, ds_name: str) -> str:
    if ds_name=="MIMIC_IV":
        try:
            parts=Path(fp_str).parts
            for p in reversed(parts):
                if p.startswith('p') and len(p)>1 and p[1:].isdigit() and len(p)>5: 
                    return p
            return fn.split('_')[0].split('.')[0] if '_' in fn else fn.split('.')[0]
        except Exception: 
            return fn.split('.')[0]
    elif ds_name == "Assignment3":
        return fn.split('.')[0]
    return fn.split('.')[0]

def process_single_file(row, dataset_name, extension):
    """Process a single file - can be run in parallel"""
    try:
        cr, fp, fn = str(row['Condition']), str(row['File Path']), str(row['File Name'])
        mc = config.CONDITION_MAPPING.get(cr)
        
        if not mc or mc not in config.TARGET_CONDITIONS:
            return {'status': 'skip'}
            
        pid = extract_patient_id(fp, fn, dataset_name)
        sfn = "".join(c if c.isalnum() or c in ('_', '-') else '_' for c in fn) + f"_{dataset_name.lower()}.npy"
        onp = config.NPY_FILES_DIR / sfn
        
        rd = {
            'npy_path': sfn,
            'original_condition': cr,
            'condition': mc,
            'dataset': dataset_name,
            config.PATIENT_ID_COLUMN: pid
        }
        rd.update({l: (1 if mc == l else 0) for l in config.TARGET_CONDITIONS})
        
        if config.SKIP_PREPROCESSING_IF_NPY_EXISTS and onp.exists():
            return {'status': 'success', 'record': rd, 'is_new': False}
            
        sdr, fs = load_ecg_data(fp, extension)
        
        if sdr is None or fs is None:
            return {'status': 'skip'}
            
        ps = preprocess_ecg_segment_for_hubert(sdr, fs, fn)
        if ps is None or ps.size == 0:
            return {'status': 'skip'}
            
        np.save(onp, ps)
        return {'status': 'success', 'record': rd, 'is_new': True}
        
    except Exception as e:
        return {'status': 'skip'}

def run_improved_preprocessing() -> Optional[pd.DataFrame]:
    """Parallelized preprocessing using multiple CPU cores"""
    print("--- Running Improved Parallel Preprocessing Step ---")
    print(f"Using {config.PARALLEL_PREPROCESSING_CORES} CPU cores for parallel preprocessing")
    
    prl, sc, pnc = [], 0, 0
    dsets = [
        {"name": "MIMIC_IV", "info_csv": config.MIMIC_IV_INFO_CSV, "ext": ".dat"},
        {"name": "Physionet_2021", "info_csv": config.PHYSIONET_2021_INFO_CSV, "ext": ".mat"}
    ]
    
    for dsi in dsets:
        dsn, icp, fext = dsi["name"], dsi["info_csv"], dsi["ext"]
        print(f"\nProcessing {dsn} dataset...")
        
        if not icp.exists():
            print(f"{dsn} info file not found: {icp}", file=sys.stderr)
            continue
            
        df = pd.read_csv(icp, low_memory=False)
        dff = df[df['File Extension'] == fext].copy() if 'File Extension' in df.columns else df.copy()
        
        if not all(c in dff.columns for c in ['Condition', 'File Path', 'File Name']):
            print(f"ERR: Missing cols in {icp}", file=sys.stderr)
            continue
        
        # Process files in parallel
        with ProcessPoolExecutor(max_workers=config.PARALLEL_PREPROCESSING_CORES) as executor:
            futures = []
            
            for _, r in dff.iterrows():
                future = executor.submit(process_single_file, r, dsn, fext)
                futures.append(future)
            
            # Process results as they complete
            for future in tqdm(as_completed(futures), total=len(futures), desc=dsn, unit="file"):
                result = future.result()
                if result is not None:
                    if result['status'] == 'success':
                        prl.append(result['record'])
                        if result['is_new']:
                            pnc += 1
                    else:
                        sc += 1
    
    print(f"\nPreproc Summary: Skip:{sc}, New:{pnc}, TotalRecs:{len(prl)}")
    
    if not prl:
        print("No records processed. Exiting.", file=sys.stderr)
        return None
        
    fdf = pd.DataFrame(prl)
    for l in config.TARGET_CONDITIONS:
        if l not in fdf.columns:
            fdf[l] = 0
            
    fdf = fdf[['npy_path', config.PATIENT_ID_COLUMN, 'dataset', 'condition', 'original_condition'] + config.TARGET_CONDITIONS]
    fdf.drop_duplicates(subset=['npy_path'], inplace=True)
    print(f"Unique records post-preproc: {len(fdf)}")
    return fdf

def process_assignment3_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Process Assignment 3 data which only has validation and test sets (unlabeled)"""
    print("\n--- Processing Assignment 3 Data ---")
    
    if not config.ASSIGNMENT3_INFO_CSV.exists():
        print(f"Assignment3 info file not found: {config.ASSIGNMENT3_INFO_CSV}")
        return pd.DataFrame(), pd.DataFrame()
    
    file_info_df = pd.read_csv(config.ASSIGNMENT3_INFO_CSV)
    
    # Note: Assignment 3 files don't have conditions in filenames
    # They will be predicted by the model
    print("Note: Assignment 3 files are unlabeled. Conditions will be predicted by the model.")
    
    val_records = []
    test_records = []
    
    for _, row in file_info_df.iterrows():
        fn = row['File Name']
        file_loc = row['File Location']
        
        # Determine if validation or test based on filename
        is_validation = 'validation' in fn.lower()
        
        # Create record without condition (will be predicted)
        pid = extract_patient_id(file_loc, fn, "Assignment3")
        sfn = f"{fn}_assignment3.npy"
        onp = config.NPY_FILES_DIR / sfn
        
        rd = {
            'npy_path': sfn,
            'original_condition': 'Unknown',  # Unknown condition
            'condition': 'Unknown',  # Will be predicted
            'dataset': 'Assignment3',
            config.PATIENT_ID_COLUMN: pid,
            'original_file_path': file_loc,
            'original_filename': fn  # Store original filename for reporting
        }
        
        # Set all labels to 0 (unknown)
        for l in config.TARGET_CONDITIONS:
            rd[l] = 0
        
        # Check if preprocessed file exists or preprocess
        if config.SKIP_PREPROCESSING_IF_NPY_EXISTS and onp.exists():
            if is_validation:
                val_records.append(rd)
            else:
                test_records.append(rd)
            print(f"  Using existing preprocessed file for {fn}")
        else:
            # Load and preprocess
            sdr, fs = load_ecg_data(file_loc, '.npy')
            if sdr is None or fs is None:
                print(f"Failed to load: {file_loc}")
                continue
            
            try:
                # For Assignment 3, data is already at 100Hz, so adjust preprocessing
                if fs == 100:
                    # Data is already at target rate, just ensure correct shape and scaling
                    ps = sdr.copy()
                    if ps.shape[1] != config.TARGET_SR_PREPROCESS * config.SEGMENT_DURATION_SECONDS:
                        # Resample to 500Hz for consistency with preprocessing
                        ps_500 = []
                        for lead in ps:
                            resampled = resample_poly(lead, 5, 1, window=('kaiser', 5.0))
                            ps_500.append(resampled)
                        ps = np.array(ps_500)
                        ps = preprocess_ecg_segment_for_hubert(ps, 500, fn)
                    else:
                        ps = preprocess_ecg_segment_for_hubert(ps, fs, fn)
                else:
                    ps = preprocess_ecg_segment_for_hubert(sdr, fs, fn)
                    
                if ps is None or ps.size == 0:
                    print(f"Preprocessing failed for: {fn}")
                    continue
                    
                np.save(onp, ps)
                print(f"  Preprocessed and saved {fn}")
                
                if is_validation:
                    val_records.append(rd)
                else:
                    test_records.append(rd)
                    
            except Exception as e:
                print(f"Error processing {fn}: {e}")
                continue
    
    val_df = pd.DataFrame(val_records) if val_records else pd.DataFrame()
    test_df = pd.DataFrame(test_records) if test_records else pd.DataFrame()
    
    print(f"Assignment3 - Validation records: {len(val_df)}, Test records: {len(test_df)}")
    if len(val_df) > 0:
        print(f"  Validation files: {', '.join(val_df['original_filename'].tolist())}")
    if len(test_df) > 0:
        print(f"  Test files: {', '.join(test_df['original_filename'].tolist())}")
    
    return val_df, test_df

# ==============================================================================
# == 5. DATA SPLITTING
# ==============================================================================
def perform_data_splits(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    print("\n--- Performing Data Splits ---")
    pat_df=df.groupby(config.PATIENT_ID_COLUMN).first().reset_index()
    lbl_strat,pids=pat_df[config.TARGET_CONDITIONS].values,pat_df[config.PATIENT_ID_COLUMN].values
    if len(pids)<3: 
        raise ValueError(f"Need >=3 unique patients for split, got {len(pids)}.")
    fst_sz=config.VAL_RATIO+config.TEST_RATIO
    if not (0<fst_sz<1.0): 
        raise ValueError("VAL_RATIO+TEST_RATIO must be >0 and <1.")
    msss1=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=fst_sz,random_state=config.RANDOM_STATE)
    try: 
        tpi,tmpi=next(msss1.split(pids,lbl_strat))
    except ValueError:
        print("WARN: Stratification fail for initial split. Using random.",file=sys.stderr)
        tpi,tmpi=train_test_split(np.arange(len(pids)),test_size=fst_sz,random_state=config.RANDOM_STATE,shuffle=True)
    tp,tmp_p,tmp_ls=pids[tpi],pids[tmpi],lbl_strat[tmpi]
    vp,testp=np.array([]),np.array([])
    if len(tmp_p)>0:
        if config.VAL_RATIO==0 and config.TEST_RATIO>0: 
            testp=tmp_p
        elif config.TEST_RATIO==0 and config.VAL_RATIO>0: 
            vp=tmp_p
        elif len(tmp_p)<2 or (config.VAL_RATIO+config.TEST_RATIO)==0:
            vp=tmp_p if config.VAL_RATIO>0 else np.array([])
            testp=tmp_p if config.TEST_RATIO>0 and config.VAL_RATIO==0 else np.array([])
        else:
            sst_sz=config.TEST_RATIO/(config.VAL_RATIO+config.TEST_RATIO)
            msss2=MultilabelStratifiedShuffleSplit(n_splits=1,test_size=sst_sz,random_state=config.RANDOM_STATE)
            try: 
                vli,tli=next(msss2.split(tmp_p,tmp_ls)); 
                vp,testp=tmp_p[vli],tmp_p[tli]
            except ValueError:
                print("WARN: Stratification fail for val/test. Using random.",file=sys.stderr)
                vli,tli=train_test_split(np.arange(len(tmp_p)),test_size=sst_sz,random_state=config.RANDOM_STATE,shuffle=True)
                vp,testp=tmp_p[vli],tmp_p[tli]
    tr_df,v_df,te_df=df[df[config.PATIENT_ID_COLUMN].isin(tp)].reset_index(drop=True),\
                     df[df[config.PATIENT_ID_COLUMN].isin(vp)].reset_index(drop=True),\
                     df[df[config.PATIENT_ID_COLUMN].isin(testp)].reset_index(drop=True)
    print(f"Pats: Tot {len(pids)},Tr {len(tp)}({len(tr_df)}r),Val {len(vp)}({len(v_df)}r),Test {len(testp)}({len(te_df)}r)")
    assert len(set(tp)&set(vp))==0 and len(set(tp)&set(testp))==0 and len(set(vp)&set(testp))==0,"Patient Leakage!"
    return tr_df,v_df,te_df

# ==============================================================================
# == 6. LOSS FUNCTIONS
# ==============================================================================
class FocalLoss(nn.Module):
    def __init__(self, alpha=config.FOCAL_LOSS_ALPHA, gamma=config.FOCAL_LOSS_GAMMA, reduction='mean'):
        super(FocalLoss,self).__init__(); 
        self.a,self.g,self.r=alpha,gamma,reduction
        
    def forward(self, i, t):
        bce=nn.BCEWithLogitsLoss(reduction='none')(i,t); 
        pt=torch.exp(-bce)
        fl=self.a*(1-pt)**self.g*bce
        return torch.mean(fl) if self.r=='mean' else (torch.sum(fl) if self.r=='sum' else fl)

# ==============================================================================
# == 7. TRAINING FUNCTION WITH HARDWARE MONITORING
# ==============================================================================
def train_model(model: HuBERTForECGClassification, train_loader: DataLoader, val_loader: DataLoader, device: torch.device):
    print("\n--- Starting Model Training ---")
    
    # Start hardware monitoring
    hardware_monitor = HardwareMonitor(log_interval=config.HARDWARE_MONITORING_INTERVAL)
    hardware_monitor.start()
    
    try:
        scaler = GradScaler(enabled=config.USE_AMP)
        criterion = None
        if config.LOSS_TYPE == "bce_with_pos_weight":
            labels_for_pos_weight = train_loader.dataset.df[config.TARGET_CONDITIONS].values
            pos_counts = np.sum(labels_for_pos_weight, axis=0) + 1e-8
            neg_counts = len(labels_for_pos_weight) - pos_counts + 1e-8
            pos_weight_values = np.clip(neg_counts / pos_counts, 1e-3, 1e3)
            pos_weight = torch.tensor(pos_weight_values, dtype=torch.float32).to(device)
            print(f"Calculated pos_weight for BCE: {pos_weight.cpu().numpy()}")
            criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        elif config.LOSS_TYPE == "focal_loss": 
            criterion = FocalLoss()
        else: 
            criterion = nn.BCEWithLogitsLoss()

        # Stage 1
        print("\n--- Stage 1: Training Classifier Head ---")
        stage1_start_time = time.time()
        
        for param in model.hubert_ecg.parameters(): 
            param.requires_grad = False
        for param in model.classifier.parameters(): 
            param.requires_grad = True
        optimizer_stage1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE_STAGE1)
        scheduler_stage1 = CosineAnnealingLR(optimizer_stage1, T_max=max(1, config.NUM_EPOCHS_STAGE1 - config.LR_SCHEDULER_WARMUP_EPOCHS), eta_min=1e-7)
        best_val_loss_stage1 = float('inf')

        for epoch in range(config.NUM_EPOCHS_STAGE1):
            epoch_start_time = time.time()
            model.train(); 
            total_train_loss_stage1 = 0; 
            optimizer_stage1.zero_grad()
            progress_bar_train = tqdm(train_loader, desc=f"S1 E{epoch+1}/{config.NUM_EPOCHS_STAGE1} [Train]", unit="b")
            for batch_idx, batch_data in enumerate(progress_bar_train):
                inputs, labels = batch_data['ecg_data'].to(device).float(), batch_data['labels'].to(device).float()
                with torch.amp.autocast(device_type=device.type, dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / config.GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    if config.CLIP_GRAD_NORM > 0: 
                        scaler.unscale_(optimizer_stage1); 
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                    scaler.step(optimizer_stage1); 
                    scaler.update(); 
                    optimizer_stage1.zero_grad()
                total_train_loss_stage1 += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
                progress_bar_train.set_postfix({'loss': f"{total_train_loss_stage1 / (batch_idx + 1):.4f}", 'lr': f"{optimizer_stage1.param_groups[0]['lr']:.2e}"})
            avg_train_loss_stage1 = total_train_loss_stage1 / len(train_loader)
            if epoch >= config.LR_SCHEDULER_WARMUP_EPOCHS: 
                scheduler_stage1.step()
            elif config.LR_SCHEDULER_WARMUP_EPOCHS > 0:
                for pg in optimizer_stage1.param_groups: 
                    pg['lr'] = config.LEARNING_RATE_STAGE1*((epoch+1)/config.LR_SCHEDULER_WARMUP_EPOCHS)

            model.eval(); 
            total_val_loss_stage1 = 0
            with torch.no_grad():
                progress_bar_val = tqdm(val_loader, desc=f"S1 E{epoch+1}/{config.NUM_EPOCHS_STAGE1} [Val]", unit="b")
                for batch_data in progress_bar_val:
                    inputs, labels = batch_data['ecg_data'].to(device).float(), batch_data['labels'].to(device).float()
                    with torch.amp.autocast(device_type=device.type, dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                        outputs = model(inputs); 
                        loss = criterion(outputs, labels)
                    total_val_loss_stage1 += loss.item()
                    progress_bar_val.set_postfix({'loss': f"{total_val_loss_stage1 / (progress_bar_val.n + 1):.4f}"})
            avg_val_loss_stage1 = total_val_loss_stage1 / len(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            print(f"S1 E{epoch+1}: Train Loss: {avg_train_loss_stage1:.4f}, Val Loss: {avg_val_loss_stage1:.4f}, Time: {epoch_time:.1f}s")
            print(f"  Batch processing speed: {len(train_loader) / epoch_time:.2f} batches/sec")
            print(f"  Samples per second: {len(train_loader) * config.BATCH_SIZE / epoch_time:.2f}")
            
            if avg_val_loss_stage1 < best_val_loss_stage1:
                best_val_loss_stage1 = avg_val_loss_stage1; 
                torch.save(model.state_dict(), config.TRAINED_MODEL_PATH)
                print(f"Saved best model (S1 VL: {best_val_loss_stage1:.4f}) to {config.TRAINED_MODEL_PATH}")

        stage1_time = time.time() - stage1_start_time
        print(f"Stage 1 completed in {stage1_time/60:.1f} minutes")

        # Stage 2
        print("\n--- Stage 2: Fine-tuning Backbone and Classifier ---")
        stage2_start_time = time.time()
        
        for param in model.hubert_ecg.parameters(): 
            param.requires_grad = True
        if config.UNFREEZE_LAYERS_COUNT < len(model.hubert_ecg.hubert_model.encoder.layers):
            for param in model.hubert_ecg.hubert_model.parameters(): 
                param.requires_grad = False
            if hasattr(model.hubert_ecg.hubert_model, 'feature_extractor'):
                for param in model.hubert_ecg.hubert_model.feature_extractor.parameters(): 
                    param.requires_grad = True
            if hasattr(model.hubert_ecg.hubert_model, 'feature_projection'):
                for param in model.hubert_ecg.hubert_model.feature_projection.parameters(): 
                    param.requires_grad = True
            if hasattr(model.hubert_ecg.hubert_model.encoder, 'pos_conv_embed'):
                 for param in model.hubert_ecg.hubert_model.encoder.pos_conv_embed.parameters(): 
                     param.requires_grad = True
            num_total_layers = len(model.hubert_ecg.hubert_model.encoder.layers)
            for i in range(num_total_layers - config.UNFREEZE_LAYERS_COUNT, num_total_layers):
                if i >= 0:
                    for param in model.hubert_ecg.hubert_model.encoder.layers[i].parameters(): 
                        param.requires_grad = True
        for param in model.classifier.parameters(): 
            param.requires_grad = True
        trainable_params_stage2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"S2 Trainable params: {trainable_params_stage2:,}")
        optimizer_stage2 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=config.LEARNING_RATE_STAGE2)
        scheduler_stage2 = CosineAnnealingLR(optimizer_stage2, T_max=max(1, config.NUM_EPOCHS_STAGE2 - config.LR_SCHEDULER_WARMUP_EPOCHS), eta_min=1e-8)
        best_val_loss_stage2 = float('inf')
        if config.TRAINED_MODEL_PATH.exists():
            print(f"Loading best model from S1: {config.TRAINED_MODEL_PATH}")
            model.load_state_dict(torch.load(config.TRAINED_MODEL_PATH, map_location=device))

        for epoch in range(config.NUM_EPOCHS_STAGE2):
            epoch_start_time = time.time()
            model.train(); 
            total_train_loss_stage2 = 0; 
            optimizer_stage2.zero_grad()
            progress_bar_train = tqdm(train_loader, desc=f"S2 E{epoch+1}/{config.NUM_EPOCHS_STAGE2} [Train]", unit="b")
            for batch_idx, batch_data in enumerate(progress_bar_train):
                inputs, labels = batch_data['ecg_data'].to(device).float(), batch_data['labels'].to(device).float()
                with torch.amp.autocast(device_type=device.type, dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels) / config.GRADIENT_ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if (batch_idx + 1) % config.GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1) == len(train_loader):
                    if config.CLIP_GRAD_NORM > 0: 
                        scaler.unscale_(optimizer_stage2); 
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.CLIP_GRAD_NORM)
                    scaler.step(optimizer_stage2); 
                    scaler.update(); 
                    optimizer_stage2.zero_grad()
                total_train_loss_stage2 += loss.item() * config.GRADIENT_ACCUMULATION_STEPS
                progress_bar_train.set_postfix({'loss': f"{total_train_loss_stage2 / (batch_idx + 1):.4f}", 'lr': f"{optimizer_stage2.param_groups[0]['lr']:.2e}"})
            avg_train_loss_stage2 = total_train_loss_stage2 / len(train_loader)
            if epoch >= config.LR_SCHEDULER_WARMUP_EPOCHS: 
                scheduler_stage2.step()
            elif config.LR_SCHEDULER_WARMUP_EPOCHS > 0:
                for pg in optimizer_stage2.param_groups: 
                    pg['lr'] = config.LEARNING_RATE_STAGE2*((epoch+1)/config.LR_SCHEDULER_WARMUP_EPOCHS)

            model.eval(); 
            total_val_loss_stage2 = 0
            with torch.no_grad():
                progress_bar_val = tqdm(val_loader, desc=f"S2 E{epoch+1}/{config.NUM_EPOCHS_STAGE2} [Val]", unit="b")
                for batch_data in progress_bar_val:
                    inputs, labels = batch_data['ecg_data'].to(device).float(), batch_data['labels'].to(device).float()
                    with torch.amp.autocast(device_type=device.type, dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                        outputs = model(inputs); 
                        loss = criterion(outputs, labels)
                    total_val_loss_stage2 += loss.item()
                    progress_bar_val.set_postfix({'loss': f"{total_val_loss_stage2 / (progress_bar_val.n + 1):.4f}"})
            avg_val_loss_stage2 = total_val_loss_stage2 / len(val_loader)
            
            epoch_time = time.time() - epoch_start_time
            print(f"S2 E{epoch+1}: Train Loss: {avg_train_loss_stage2:.4f}, Val Loss: {avg_val_loss_stage2:.4f}, Time: {epoch_time:.1f}s")
            print(f"  Batch processing speed: {len(train_loader) / epoch_time:.2f} batches/sec")
            print(f"  Samples per second: {len(train_loader) * config.BATCH_SIZE / epoch_time:.2f}")
            
            if avg_val_loss_stage2 < best_val_loss_stage2:
                best_val_loss_stage2 = avg_val_loss_stage2; 
                torch.save(model.state_dict(), config.TRAINED_MODEL_PATH)
                print(f"Saved best model (S2 VL: {best_val_loss_stage2:.4f}) to {config.TRAINED_MODEL_PATH}")

        stage2_time = time.time() - stage2_start_time
        print(f"Stage 2 completed in {stage2_time/60:.1f} minutes")
        print(f"Total training time: {(stage1_time + stage2_time)/60:.1f} minutes")

        print("Model training complete!")
        if config.TRAINED_MODEL_PATH.exists():
            print(f"Loading best overall trained model from {config.TRAINED_MODEL_PATH}")
            model.load_state_dict(torch.load(config.TRAINED_MODEL_PATH, map_location=device))
        
        return model
        
    finally:
        # Stop hardware monitoring
        hardware_monitor.stop()

# ==============================================================================
# == 8. THRESHOLD OPTIMIZATION
# ==============================================================================
def optimize_thresholds(model: HuBERTForECGClassification, val_loader: DataLoader, device: torch.device) -> Dict[str, float]:
    print("\n--- Optimizing Decision Thresholds ---")
    model.eval(); 
    all_val_probas, all_val_labels = [], []
    with torch.no_grad():
        for batch_data in tqdm(val_loader, desc="Optimizing Thresholds (Val inference)", unit="b"):
            inputs, labels = batch_data['ecg_data'].to(device).float(), batch_data['labels'].to(device).float()
            with torch.amp.autocast(device_type=device.type, dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                outputs = model(inputs)
            all_val_probas.append(torch.sigmoid(outputs).cpu()); 
            all_val_labels.append(labels.cpu())
    all_val_probas_np, all_val_labels_np = torch.cat(all_val_probas).numpy(), torch.cat(all_val_labels).numpy()
    optimized_thresholds = {}
    print(f"Optimizing for F-beta score with beta = {config.FBETA_BETA}")
    for i, cond_name in enumerate(config.TARGET_CONDITIONS):
        probas_class, labels_class = all_val_probas_np[:, i], all_val_labels_np[:, i]
        best_thresh, best_fbeta = 0.5, -1.0
        if np.sum(labels_class) == 0:
            print(f"Warning: No positive samples for '{cond_name}' in val set. Default threshold 0.5.")
            optimized_thresholds[cond_name] = 0.5; 
            continue
        for thresh_cand in np.linspace(0.01, 0.99, 99):
            preds_class = (probas_class >= thresh_cand).astype(int)
            current_fbeta = fbeta_score(labels_class, preds_class, beta=config.FBETA_BETA, zero_division=0)
            if current_fbeta > best_fbeta: 
                best_fbeta, best_thresh = current_fbeta, thresh_cand
        optimized_thresholds[cond_name] = best_thresh
        print(f"  {cond_name}: OptimalThresh={best_thresh:.3f} (F{config.FBETA_BETA}-score={best_fbeta:.4f})")
    with open(config.OPTIMIZED_THRESHOLDS_PATH, 'w') as f: 
        json.dump(optimized_thresholds, f, indent=4)
    print(f"Optimized thresholds saved to {config.OPTIMIZED_THRESHOLDS_PATH}")
    return optimized_thresholds

# ==============================================================================
# == 9. EVALUATION FUNCTIONS
# ==============================================================================
def calculate_metrics(y_true, y_pred, y_proba=None, condition_name="Unknown"):
    if isinstance(y_true, torch.Tensor): y_true = y_true.cpu().numpy()
    if isinstance(y_pred, torch.Tensor): y_pred = y_pred.cpu().numpy()
    if isinstance(y_proba, torch.Tensor): y_proba = y_proba.cpu().numpy()
    metrics_dict = {'sensitivity': 0.0, 'specificity': 0.0, 'precision': 0.0,
                    'f1': 0.0, 'auc': np.nan, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}
    try:
        if y_pred.dtype != np.int_ and len(np.unique(y_pred)) > 2 :
             y_pred = (y_pred >= 0.5).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0,1])
        tn, fp, fn, tp = cm.ravel()
        metrics_dict.update({'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)})
        if (tp + fn) > 0: metrics_dict['sensitivity'] = tp / (tp + fn)
        if (tn + fp) > 0: metrics_dict['specificity'] = tn / (tn + fp)
        if (tp + fp) > 0: metrics_dict['precision'] = tp / (tp + fp)
        prec_val, sens_val = metrics_dict['precision'], metrics_dict['sensitivity']
        if (prec_val + sens_val) > 0: metrics_dict['f1'] = 2 * (prec_val * sens_val) / (prec_val + sens_val)
        if y_proba is not None and len(np.unique(y_true)) > 1:
            try: 
                metrics_dict['auc'] = roc_auc_score(y_true, y_proba)
            except ValueError: 
                pass
    except Exception as e: 
        print(f"Error in calculate_metrics for {condition_name}: {e}", file=sys.stderr)
    return metrics_dict

def run_evaluation(model: HuBERTForECGClassification, dataloader: DataLoader, device: torch.device,
                   optimized_thresholds: Dict[str, float], dataset_name_for_report: str = "Test_Set", 
                   export_confidence_scores: bool = True):
    print(f"\n--- Running Evaluation on {dataset_name_for_report} ---")
    model.eval(); 
    all_probas_list, all_labels_list, all_sample_info_list = [], [], []
    
    eval_start_time = time.time()
    
    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc=f"Evaluating ({dataset_name_for_report})", unit="b"):
            inputs, labels_batch = batch_data['ecg_data'].to(device).float(), batch_data['labels'].to(device).float()
            with torch.amp.autocast(device_type=device.type, dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                outputs_logits = model(inputs)
            probas_batch = torch.sigmoid(outputs_logits)
            all_probas_list.append(probas_batch.cpu()); 
            all_labels_list.append(labels_batch.cpu())
            for i in range(inputs.size(0)):
                all_sample_info_list.append({'npy_path': batch_data['npy_path'][i],
                                             'actual_condition_str': batch_data['condition_str'][i],
                                             'dataset_source': batch_data['dataset_str'][i],
                                             'probas': probas_batch[i].cpu().numpy(),
                                             'true_labels': labels_batch[i].cpu().numpy().astype(int)})
    
    eval_time = time.time() - eval_start_time
    print(f"Evaluation inference completed in {eval_time:.1f}s ({len(dataloader) * config.BATCH_SIZE / eval_time:.1f} samples/sec)")
    
    all_probas_np, all_labels_np = torch.cat(all_probas_list).numpy(), torch.cat(all_labels_list).numpy().astype(int)
    all_preds_np = np.zeros_like(all_probas_np, dtype=int)
    for i, cond_name in enumerate(config.TARGET_CONDITIONS):
        all_preds_np[:, i] = (all_probas_np[:, i] >= optimized_thresholds.get(cond_name, 0.5)).astype(int)

    print(f"\n=== {dataset_name_for_report} EVALUATION RESULTS ===")
    eval_summary_records = []
    for i, cond_name in enumerate(config.TARGET_CONDITIONS):
        true_labels, pred_labels, proba_scores = all_labels_np[:, i], all_preds_np[:, i], all_probas_np[:, i]
        metrics = calculate_metrics(true_labels, pred_labels, proba_scores, cond_name)
        num_total, num_pos = len(true_labels), int(np.sum(true_labels))
        record = {'Dataset_Split': dataset_name_for_report, 'Condition': cond_name,
                  'Total_Samples': num_total, 'Positive_Samples': num_pos,
                  'Threshold': optimized_thresholds.get(cond_name, 0.5), **metrics}
        eval_summary_records.append(record)
        auc_display = record['auc'] if not pd.isna(record['auc']) else 'N/A'
        print(f"\n{cond_name}: (Thresh: {record['Threshold']:.3f}) Samples (Tot/Pos): {num_total}/{num_pos}")
        print(f"  TP:{metrics['tp']} FP:{metrics['fp']} TN:{metrics['tn']} FN:{metrics['fn']}")
        print(f"  AUC: {auc_display if isinstance(auc_display, str) else f'{auc_display:.4f}'}, Sens: {metrics['sensitivity']:.4f}, Spec: {metrics['specificity']:.4f}, Prec: {metrics['precision']:.4f}, F1: {metrics['f1']:.4f}")
        if len(np.unique(true_labels)) > 1 :
            cm = confusion_matrix(true_labels, pred_labels, labels=[0,1])
            plt.figure(figsize=(7, 5)); 
            sns.set_theme(style="whitegrid")
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                       xticklabels=['Pred Neg', 'Pred Pos'], yticklabels=['Actual Neg', 'Actual Pos'])
            plt.title(f'CM - {dataset_name_for_report} - {cond_name}', fontsize=12)
            plt.ylabel('Actual Label', fontsize=10); 
            plt.xlabel('Predicted Label', fontsize=10)
            plt.tight_layout(); 
            plt.savefig(config.CONFUSION_MATRICES_DIR / f"CM_{dataset_name_for_report}_{cond_name}.png", dpi=150); 
            plt.close()
    
    df_results = pd.DataFrame(eval_summary_records)
    if 'auc' in df_results.columns:
        df_results['auc'] = df_results['auc'].apply(lambda x: 'N/A' if pd.isna(x) else x)
    else: 
        print("Warning: 'auc' column not found in df_results for CSV processing.", file=sys.stderr)

    df_results.to_csv(config.RESULTS_CSV_PATH, mode='a', header=not config.RESULTS_CSV_PATH.exists(), index=False)
    print(f"📊 Eval results for {dataset_name_for_report} appended to: {config.RESULTS_CSV_PATH}")
    
    # Export individual predictions with confidence scores
    indiv_preds_rows = []
    confidence_scores_rows = []
    for sample_data in all_sample_info_list:
        row = {'Split': dataset_name_for_report, 'NPY_Path': sample_data['npy_path'],
               'Actual_Condition_String': sample_data['actual_condition_str'], 'Dataset_Source': sample_data['dataset_source']}
        conf_row = row.copy()
        for i, cond_name in enumerate(config.TARGET_CONDITIONS):
            row[f'True_{cond_name}'] = sample_data['true_labels'][i]
            row[f'Proba_{cond_name}'] = f"{sample_data['probas'][i]:.4f}"
            row[f'Pred_{cond_name}'] = 1 if sample_data['probas'][i] >= optimized_thresholds.get(cond_name, 0.5) else 0
            conf_row[f'Confidence_{cond_name}_%'] = f"{sample_data['probas'][i] * 100:.2f}"
        indiv_preds_rows.append(row)
        confidence_scores_rows.append(conf_row)
        
    pd.DataFrame(indiv_preds_rows).to_csv(config.SAMPLE_PREDICTIONS_CSV_PATH, mode='a', header=not config.SAMPLE_PREDICTIONS_CSV_PATH.exists(), index=False)
    print(f"📄 Sample preds for {dataset_name_for_report} appended to: {config.SAMPLE_PREDICTIONS_CSV_PATH}")
    
    if export_confidence_scores:
        pd.DataFrame(confidence_scores_rows).to_csv(config.CONFIDENCE_SCORES_CSV_PATH, mode='a', 
                                                   header=not config.CONFIDENCE_SCORES_CSV_PATH.exists(), index=False)
        print(f"📄 Confidence scores for {dataset_name_for_report} appended to: {config.CONFIDENCE_SCORES_CSV_PATH}")
    
    return df_results

def evaluate_assignment3_data(model: HuBERTForECGClassification, val_loader: DataLoader, test_loader: DataLoader, 
                             device: torch.device, optimized_thresholds: Dict[str, float]):
    """Special evaluation for Assignment 3 data with detailed predictions (unlabeled data)"""
    print("\n--- Evaluating Assignment 3 Data (Unlabeled) ---")
    print("Note: Assignment 3 data is unlabeled. Showing predictions only.")
    
    assignment3_predictions = []
    
    for loader, split_name in [(val_loader, "Assignment3_Val"), (test_loader, "Assignment3_Test")]:
        if loader is None:
            continue
            
        model.eval()
        with torch.no_grad():
            for batch_data in tqdm(loader, desc=f"Predicting {split_name}", unit="b"):
                inputs = batch_data['ecg_data'].to(device).float()
                npy_paths = batch_data['npy_path']
                
                with torch.amp.autocast(device_type=device.type, dtype=config.AMP_DTYPE, enabled=config.USE_AMP):
                    outputs_logits = model(inputs)
                probas = torch.sigmoid(outputs_logits).cpu().numpy()
                
                for i in range(inputs.size(0)):
                    # Find predicted condition with highest confidence
                    max_conf_idx = np.argmax(probas[i])
                    predicted_condition = config.TARGET_CONDITIONS[max_conf_idx]
                    max_confidence = probas[i][max_conf_idx] * 100
                    
                    # Get all confidence scores
                    all_confidences = {cond: probas[i][j] * 100 for j, cond in enumerate(config.TARGET_CONDITIONS)}
                    
                    # Sort conditions by confidence for better display
                    sorted_conditions = sorted(all_confidences.items(), key=lambda x: x[1], reverse=True)
                    
                    # Get filename from npy_path
                    original_filename = npy_paths[i].replace('_assignment3.npy', '')
                    
                    assignment3_predictions.append({
                        'Split': split_name,
                        'File_Name': original_filename,
                        'Predicted_Condition': predicted_condition,
                        'Confidence_%': f"{max_confidence:.2f}",
                        'Top_3_Predictions': ', '.join([f"{cond}: {conf:.1f}%" for cond, conf in sorted_conditions[:3]]),
                        **{f'{cond}_Confidence_%': f"{conf:.2f}" for cond, conf in all_confidences.items()}
                    })
    
    # Save Assignment 3 predictions
    if assignment3_predictions:
        df_assignment3 = pd.DataFrame(assignment3_predictions)
        df_assignment3.to_csv(config.ASSIGNMENT3_PREDICTIONS_CSV_PATH, index=False)
        print(f"\n📄 Assignment 3 predictions saved to: {config.ASSIGNMENT3_PREDICTIONS_CSV_PATH}")
        
        # Print detailed summary
        print("\n=== Assignment 3 Predictions Summary ===")
        print("File Name | Split | Predicted Condition | Confidence | Top 3 Predictions")
        print("-" * 100)
        for _, row in df_assignment3.iterrows():
            print(f"{row['File_Name']:15} | {row['Split']:15} | {row['Predicted_Condition']:25} | {row['Confidence_%']:>7}% | {row['Top_3_Predictions']}")
        
        # Print condition distribution
        print("\n=== Predicted Condition Distribution ===")
        condition_counts = df_assignment3['Predicted_Condition'].value_counts()
        for condition, count in condition_counts.items():
            print(f"{condition}: {count} files ({count/len(df_assignment3)*100:.1f}%)")
    else:
        print("No Assignment 3 predictions to save.")

# ==============================================================================
# == 10. REPORTING DATA SPLIT STATISTICS
# ==============================================================================
def report_data_split_statistics(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame, 
                                assignment3_val_df: pd.DataFrame = None, assignment3_test_df: pd.DataFrame = None):
    print("\n--- Reporting Data Split Statistics ---")
    report_data = []
    splits = {'Train': train_df, 'Validation': val_df, 'Test': test_df}
    
    if assignment3_val_df is not None and not assignment3_val_df.empty:
        splits['Assignment3_Val'] = assignment3_val_df
    if assignment3_test_df is not None and not assignment3_test_df.empty:
        splits['Assignment3_Test'] = assignment3_test_df
    
    total_overall_samples = sum(len(df) for df in splits.values() if not df.empty)
    if total_overall_samples == 0: 
        print("No data in any split."); 
        return

    for split_name, df_split in splits.items():
        if df_split.empty: 
            print(f"{split_name} set is empty."); 
            continue
        num_recs_split = len(df_split)
        num_pats_split = df_split[config.PATIENT_ID_COLUMN].nunique()
        print(f"\n{split_name} Set: Records: {num_recs_split} ({num_recs_split/total_overall_samples*100:.1f}%), Patients: {num_pats_split}")
        for ds_src_name in df_split['dataset'].unique():
            df_s_ds = df_split[df_split['dataset'] == ds_src_name]
            num_recs_s_ds = len(df_s_ds)
            print(f"  Dataset: {ds_src_name} - Records: {num_recs_s_ds}")
            
            # Special handling for Assignment3 (unlabeled data)
            if ds_src_name == 'Assignment3':
                print(f"    Note: Assignment3 data is unlabeled (conditions to be predicted)")
                continue
            
            for cond_col in config.TARGET_CONDITIONS:
                num_pos = df_s_ds[cond_col].sum()
                perc_pos = (num_pos/num_recs_s_ds*100) if num_recs_s_ds > 0 else 0
                report_data.append({'Split': split_name, 'Dataset_Source': ds_src_name, 'Condition': cond_col,
                                    'Num_Positive': num_pos, 'Perc_Positive': f"{perc_pos:.1f}%",
                                    'Total_In_Split_Dataset': num_recs_s_ds, 'Total_In_Split': num_recs_split,
                                    'Total_Patients_In_Split': num_pats_split})
    pd.DataFrame(report_data).to_csv(config.DATA_SPLIT_REPORT_PATH, index=False)
    print(f"\nData split statistics saved to: {config.DATA_SPLIT_REPORT_PATH}")

# ==============================================================================
# == 11. OPTIMIZED DATA LOADER CREATION
# ==============================================================================
def create_optimized_dataloader(dataset, batch_size, shuffle=False, is_train=False, num_workers=None):
    """Create DataLoader with RTX 5090 + 128GB RAM optimizations"""
    if num_workers is None:
        num_workers = config.NUM_WORKERS
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,  # Pin memory for faster GPU transfer
        persistent_workers=num_workers > 0,
        prefetch_factor=config.PREFETCH_FACTOR if num_workers > 0 else None,  # Increased to 4
        drop_last=is_train,  # Drop incomplete batches during training
    )

# ==============================================================================
# == 12. MAIN EXECUTION
# ==============================================================================
if __name__ == "__main__":
    print(f"=== HuBERT-ECG Advanced Training & Evaluation Pipeline (PID: {os.getpid()}) ===")
    print(f"=== Hardware-Optimized for Ryzen 9 7900X + RTX 5090 + 128GB RAM ===")
    
    # Display system information
    if HARDWARE_MONITORING_AVAILABLE:
        import psutil
        memory = psutil.virtual_memory()
        print(f"\n💻 System Information:")
        print(f"   CPU: {multiprocessing.cpu_count()} cores")
        print(f"   RAM: {memory.total/(1024**3):.1f} GB total, {memory.available/(1024**3):.1f} GB available")
        print(f"   RAM Usage: {memory.percent:.1f}% ({memory.used/(1024**3):.1f} GB used)")
    
    # Enable PyTorch optimizations
    if config.ENABLE_CUDNN_BENCHMARK:
        torch.backends.cudnn.benchmark = True
        print("\n✓ Enabled cuDNN benchmark mode for GPU optimization")
    
    if config.ENABLE_TF32 and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("✓ Enabled TF32 for faster matrix operations on RTX 5090")
    
    # Set thread count for CPU operations
    torch.set_num_threads(multiprocessing.cpu_count())
    print(f"✓ Set PyTorch CPU threads to {multiprocessing.cpu_count()}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device} (type: {device.type})")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"CUDA: {torch.version.cuda}")
        print(f"BF16 Support: {torch.cuda.is_bf16_supported()}")
    
    print(f"\n📁 Output directory: {config.OUTPUT_BASE_DIR}")
    print(f"📁 Preprocessed files directory: {config.PREPROCESS_FILES_DIR}")
    
    print(f"\n⚙️  Optimization Settings:")
    print(f"   Batch Size: {config.BATCH_SIZE}")
    print(f"   Workers: {config.NUM_WORKERS}")
    print(f"   Preprocessing Cores: {config.PARALLEL_PREPROCESSING_CORES}")
    print(f"   Prefetch Factor: {config.PREFETCH_FACTOR}")
    print(f"   Cache in Memory: {config.CACHE_DATASET_IN_MEMORY}")
    
    if config.NUM_WORKERS > 0 and os.name == 'nt' and sys.version_info.major == 3 and sys.version_info.minor >= 8 :
        try:
            if torch.multiprocessing.get_start_method(allow_none=True) != 'spawn':
                torch.multiprocessing.set_start_method('spawn', force=True)
                print("Set torch multiprocessing start method to 'spawn'.")
        except RuntimeError as e:
            print(f"Warning: Could not set multiprocessing start method to 'spawn': {e}. May already be set or not applicable.")

    # Clear previous results
    if config.RESULTS_CSV_PATH.exists(): os.remove(config.RESULTS_CSV_PATH)
    if config.SAMPLE_PREDICTIONS_CSV_PATH.exists(): os.remove(config.SAMPLE_PREDICTIONS_CSV_PATH)
    if config.CONFIDENCE_SCORES_CSV_PATH.exists(): os.remove(config.CONFIDENCE_SCORES_CSV_PATH)
    if config.ASSIGNMENT3_PREDICTIONS_CSV_PATH.exists(): os.remove(config.ASSIGNMENT3_PREDICTIONS_CSV_PATH)

    # Process main datasets
    preprocess_start_time = time.time()
    all_processed_df = run_improved_preprocessing()
    if all_processed_df is None or all_processed_df.empty:
        print("Preprocessing failed. Exiting.", file=sys.stderr); 
        sys.exit(1)
    preprocess_time = time.time() - preprocess_start_time
    print(f"\nPreprocessing completed in {preprocess_time/60:.1f} minutes")
    print(f"Total unique records post-preprocessing: {len(all_processed_df)}")
    print(f"Total unique patients post-preprocessing: {all_processed_df[config.PATIENT_ID_COLUMN].nunique()}")

    # Process Assignment 3 data
    print("\n=== Processing Assignment 3 Data ===")
    print("Note: Assignment 3 files (test01-06, validation01-06) are unlabeled.")
    print("The model will predict conditions for these files after training.")
    assignment3_val_df, assignment3_test_df = process_assignment3_data()

    # Perform data splits on main datasets
    train_df, val_df, test_df = perform_data_splits(all_processed_df)
    if train_df.empty or val_df.empty :
        print("Train or Validation data is empty. Exiting.", file=sys.stderr); 
        sys.exit(1)
    
    # Report statistics including Assignment 3
    report_data_split_statistics(train_df, val_df, test_df, assignment3_val_df, assignment3_test_df)

    print("\n=== Step 3: Creating Optimized DataLoaders ===")
    augmentor = ECGAugmentations() if config.AUGMENTATION_ENABLED else None
    
    # Use cached datasets for better performance
    DatasetClass = CachedECGDataset if config.CACHE_DATASET_IN_MEMORY else ECGDataset
    
    train_ds = DatasetClass(train_df, config.TARGET_CONDITIONS, config.NPY_FILES_DIR, is_train=True, augmentations=augmentor)
    val_ds = DatasetClass(val_df, config.TARGET_CONDITIONS, config.NPY_FILES_DIR, is_train=False)
    
    train_loader = create_optimized_dataloader(train_ds, config.PHYSICAL_BATCH_SIZE, shuffle=True, is_train=True)
    val_loader = create_optimized_dataloader(val_ds, config.PHYSICAL_BATCH_SIZE, shuffle=False, is_train=False)
    
    test_loader = None
    if not test_df.empty:
        test_ds = DatasetClass(test_df, config.TARGET_CONDITIONS, config.NPY_FILES_DIR, is_train=False)
        test_loader = create_optimized_dataloader(test_ds, config.PHYSICAL_BATCH_SIZE, shuffle=False, is_train=False)
    else: 
        print("Warning: Test set is empty. Final evaluation on test set will be skipped.")

    # Create Assignment 3 data loaders
    assignment3_val_loader = None
    assignment3_test_loader = None
    if not assignment3_val_df.empty:
        assignment3_val_ds = DatasetClass(assignment3_val_df, config.TARGET_CONDITIONS, config.NPY_FILES_DIR, is_train=False)
        assignment3_val_loader = create_optimized_dataloader(assignment3_val_ds, config.PHYSICAL_BATCH_SIZE, shuffle=False, is_train=False)
    if not assignment3_test_df.empty:
        assignment3_test_ds = DatasetClass(assignment3_test_df, config.TARGET_CONDITIONS, config.NPY_FILES_DIR, is_train=False)
        assignment3_test_loader = create_optimized_dataloader(assignment3_test_ds, config.PHYSICAL_BATCH_SIZE, shuffle=False, is_train=False)

    print("\n=== Step 4: Initializing Model ===")
    base_hubert_config = HuBERTECGConfig.from_pretrained(config.HUGGINGFACE_MODEL_NAME, trust_remote_code=True)
    base_ssl_model = HuBERTECG.from_pretrained(config.HUGGINGFACE_MODEL_NAME, config_obj=base_hubert_config, 
                                               ignore_mismatched_sizes=True, trust_remote_code=True)
    model = HuBERTForECGClassification(base_ssl_model, config.NUM_LABELS).to(device)
    
    # Print model size
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    print("\n=== Step 5: Training Model ===")
    training_start_time = time.time()
    trained_model = train_model(model, train_loader, val_loader, device)
    total_training_time = time.time() - training_start_time
    print(f"\n🏁 Total training completed in {total_training_time/60:.1f} minutes")

    if config.TRAINED_MODEL_PATH.exists():
        print(f"Loading best model from {config.TRAINED_MODEL_PATH} for threshold optimization.")
        trained_model.load_state_dict(torch.load(config.TRAINED_MODEL_PATH, map_location=device))
    else: 
        print(f"Warning: No trained model found at {config.TRAINED_MODEL_PATH}. Threshold optimization may not be effective.", file=sys.stderr)
    
    if not val_loader.dataset.df.empty:
        optimized_thresholds = optimize_thresholds(trained_model, val_loader, device)
    else:
        print("Validation set is empty. Skipping threshold optimization. Using default 0.5 for all classes.", file=sys.stderr)
        optimized_thresholds = {cond_name: 0.5 for cond_name in config.TARGET_CONDITIONS}

    # Evaluate on all sets
    print("\n=== Step 7: Comprehensive Evaluation ===")
    if config.TRAINED_MODEL_PATH.exists():
        print(f"Loading best model from {config.TRAINED_MODEL_PATH} for evaluation.")
        trained_model.load_state_dict(torch.load(config.TRAINED_MODEL_PATH, map_location=device))
    
    # Evaluate training set
    print("\n--- Evaluating Training Set ---")
    run_evaluation(trained_model, train_loader, device, optimized_thresholds, "Train_Set", export_confidence_scores=True)
    
    # Evaluate validation set
    print("\n--- Evaluating Validation Set ---")
    run_evaluation(trained_model, val_loader, device, optimized_thresholds, "Val_Set", export_confidence_scores=True)
    
    # Evaluate test set
    if test_loader:
        print("\n--- Evaluating Test Set ---")
        run_evaluation(trained_model, test_loader, device, optimized_thresholds, "Test_Set", export_confidence_scores=True)
    else: 
        print("\nSkipping test set evaluation as it's empty.")
    
    # Evaluate Assignment 3 data
    if assignment3_val_loader or assignment3_test_loader:
        evaluate_assignment3_data(trained_model, assignment3_val_loader, assignment3_test_loader, device, optimized_thresholds)
        
        if assignment3_val_loader:
            run_evaluation(trained_model, assignment3_val_loader, device, optimized_thresholds, 
                          "Assignment3_Val", export_confidence_scores=True)
        if assignment3_test_loader:
            run_evaluation(trained_model, assignment3_test_loader, device, optimized_thresholds, 
                          "Assignment3_Test", export_confidence_scores=True)

    final_message_title = "HuBERT ECG Process Finished!"
    final_message_body = f"All tasks complete. Results are in {config.OUTPUT_BASE_DIR}"
    print(f"\n🎉 {final_message_title}")
    print(f"📁 Results: {config.OUTPUT_BASE_DIR}")
    print(f"📄 Detailed Results: {config.RESULTS_CSV_PATH}")
    print(f"📄 Confidence Scores: {config.CONFIDENCE_SCORES_CSV_PATH}")
    print(f"📄 Assignment3 Predictions: {config.ASSIGNMENT3_PREDICTIONS_CSV_PATH}")
    
    # Assignment 3 note
    if assignment3_val_loader or assignment3_test_loader:
        print("\n📌 Note: Assignment 3 files were unlabeled. The model predicted conditions for each file.")
        print("   Check assignment3_predictions.csv for detailed predictions with confidence scores.")
    
    total_time = time.time() - preprocess_start_time
    print(f"\n⏱️  Total execution time: {total_time/60:.1f} minutes")
    print(f"   - Preprocessing: {preprocess_time/60:.1f} minutes")
    print(f"   - Training: {total_training_time/60:.1f} minutes")
    print(f"   - Evaluation: {(total_time - preprocess_time - total_training_time)/60:.1f} minutes")

    print(f"\n--- Script finished at {pd.Timestamp.now()} ---")