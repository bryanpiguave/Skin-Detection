import pandas as pd
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d
import shutil
import cv2 
def create_classifier_timestamps(csv_path):
    classifier_timestamps = pd.read_csv(csv_path)
    classifier_timestamps["signal"] =np.where(classifier_timestamps["signal"]<0.99,0,classifier_timestamps["signal"])
    classifier_timestamps["signal"] =np.where(classifier_timestamps["signal"]>=0.99,1,classifier_timestamps["signal"])

    return classifier_timestamps


def create_detector_timestamps(csv_path):
    detector_timestamps = pd.read_csv(csv_path)
    if "diff" in list(detector_timestamps.keys()):
        detector_timestamps["diff"] = (detector_timestamps["diff"] - detector_timestamps["diff"].min()) / (detector_timestamps["diff"].max() - detector_timestamps["diff"].min())
    detector_timestamps["signal"] = (detector_timestamps["Area"] - detector_timestamps["Area"].min()) / (detector_timestamps["Area"].max() - detector_timestamps["Area"].min())
    detector_timestamps["signal"] =np.where(detector_timestamps["signal"]<0.5,0,detector_timestamps["signal"])
    detector_timestamps["signal"] =np.where(detector_timestamps["signal"]>=0.5,1,detector_timestamps["signal"])

    return detector_timestamps


def create_delayed_df(fps,csv_path,reference_time,end_time,ref_df):
    delayed_timestamps = pd.read_csv(csv_path)

    # Making it flexible for different keys
    if "FECHA" in delayed_timestamps.keys():
        time_key = "FECHA"
    else:
        time_key = "time"
    delayed_timestamps[time_key]=delayed_timestamps[time_key].astype("datetime64[ns]")
    delayed_timestamps= delayed_timestamps.sort_values(by=time_key)
    delayed_timestamps = delayed_timestamps.loc[delayed_timestamps[time_key]>=reference_time]
    delayed_timestamps = delayed_timestamps.loc[delayed_timestamps[time_key]< end_time ]
    #Obtaining the frame number for each transaction
    delayed_timestamps["video_frame"] = (delayed_timestamps[time_key] - ref_df["time"][0]).dt.total_seconds()
    delayed_timestamps["video_frame"] *=fps
    delayed_timestamps["video_frame"] = delayed_timestamps["video_frame"].astype(int)
    #We defined a signal for each transaction
    delayed_timestamps["signal"]=1
    delayed_timestamps = delayed_timestamps.drop_duplicates(subset =time_key)
    return delayed_timestamps


def making_frame_window(df:pd.DataFrame,window_size:int,total_frames:int):
    transaction_signal = np.zeros(total_frames)
    for i,row in df.iterrows():
        frame = row["video_frame"]
        if i <window_size: 
            continue
        else:
            transaction_signal[int(frame-window_size):int(frame+window_size)] = 1          
            first_frames=np.arange(int(frame-window_size),int(frame))
            second_frames=np.arange(int(frame+1),int(frame+window_size+1))
            sku = row["DS_ESTADISTICO"]
            assert len(first_frames) == len(second_frames)
            lista1 =[sku]*len(first_frames)
            df1 = {'video_frame': first_frames, "signal":np.ones(len(first_frames)),"DS_ESTADISTICO":lista1}
            df2={'video_frame': second_frames, "signal":np.ones(len(second_frames)),"DS_ESTADISTICO":lista1}
        df2 =pd.DataFrame(df2)
        df1 =pd.DataFrame(df1)
        df = pd.concat([df, df1],ignore_index=True)
        df = pd.concat([df, df2],ignore_index=True)
    list_frames= list(df["video_frame"].astype(int))
    for i in range(total_frames):
        if i not in list_frames:
            df3 = {'video_frame': i, "signal":[0]}
            df3 =pd.DataFrame(df3)
            df = pd.concat([df, df3],ignore_index=True)
    df["video_frame"]=df["video_frame"].astype(int)        
    df=df.sort_values(by="video_frame",ignore_index=True)
    return transaction_signal,df



def create_signal_vector(total_frames,df: pd.DataFrame,filter=True) -> np.array:
    signal_array = np.zeros(total_frames)
    for i,row in df.iterrows():
            frame = row["video_frame"]
            signal_array[int(frame-1)] = row["signal"]
    if filter:
        signal_array = gaussian_filter1d(signal_array,2)
    return signal_array
    
def erase_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as e:
        print ("Error: %s - %s." % (e.filename, e.strerror))
    return None


def get_video_specs(video_path: str):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return total_frames,fps

def combine_signal(total_frames:int,fixed_signal:np.array,query_signal:np.array) -> np.array:
    new_combined_detector = np.zeros(total_frames)
    for i in range(total_frames):
        if fixed_signal[i]*query_signal[i] == 1 :
            new_combined_detector[i] = 1
        elif fixed_signal[i]*query_signal[i] ==0 :
            new_combined_detector[i] = 0
        elif fixed_signal[i]>query_signal[i]:
            new_combined_detector[i] = 1
        else:
            new_combined_detector[i] = 0
    return new_combined_detector

def diff_signal(total_frames:int,fixed_signal:np.array,query_signal:np.array) -> np.array:
    new_combined_detector = np.zeros(total_frames)
    for i in range(total_frames):
        if fixed_signal[i]<query_signal[i] :
            new_combined_detector[i] = 0
        elif fixed_signal[i]>query_signal[i]:
            new_combined_detector[i] = 1
        else:
            new_combined_detector[i] = fixed_signal[i]*query_signal[i]
    return new_combined_detector