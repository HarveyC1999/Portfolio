# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:46:14 2024

@author: HarveyC
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import torch
#import sys
import pandas as pd
import numpy as np
from glob import glob

import whisper
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

import re
import cn2an
import cx_Oracle

import shutil
import subprocess

import datetime
today = datetime.date.today().strftime('%Y%m%d')

import time

from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import warnings

warnings.filterwarnings('ignore')
from StarCC import PresetConversion
convert = PresetConversion(src='cn', dst='tw', with_phrase=True)

from noisereduce.torchgate import TorchGate as TG
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
import librosa
import librosa.display
import numpy as np
import soundfile as sf
import sys
from correcting import *
import gc
# In[2]:


# 記錄工作階段到log.txt檔案
def work_log(log_message):
    log_path = os.path.join(log_folder, log_file)
    with open(log_path, 'a') as f:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{current_time}] {log_message}\n')


# 記錄檔案資訊到log.txt檔案
def item_log(file_name):
    log_path = os.path.join(log_folder, log_file)
    with open(log_path, 'a') as f:
        current_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        f.write(f'[{current_time}] File: {file_name},轉檔開始\n')

# 將檔案移至 "finished" 資料夾
def move_to_finished_folder(file_name):
    source_path = os.path.join(data_folder, file_name)
    target_path = os.path.join(finished_folder, file_name)
    shutil.move(source_path, target_path)
# 降噪
def denoise(input_file):
    # 使用 librosa 讀取音訊文件
    data, rate = librosa.load(input_file, sr=None)
    
    # 確保數據是單聲道
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
    
    # 正規化音訊數據
    data = librosa.util.normalize(data)
    
    # 轉換為 PyTorch 張量
    data_tensor = torch.FloatTensor(data).unsqueeze(0).to(device)
    
    # 計算最小的 time_mask_smooth_ms
    min_time_mask_smooth_ms = max(64, int(1000 * len(data) / rate / 100))  # 至少64ms，或音頻長度的1%
    
    # 創建 TorchGating 實例，調整參數
    tg = TG(sr=rate, 
            nonstationary=True, 
            n_fft=2048, 
            prop_decrease=0.95, 
            freq_mask_smooth_hz=500,
            time_mask_smooth_ms=min_time_mask_smooth_ms).to(device)
    
    # 應用頻譜閘門到有噪音的語音信號
    enhanced_speech = tg(data_tensor)

    # 將結果轉回 NumPy 數組
    enhanced_speech_np = enhanced_speech.squeeze().cpu().numpy()
    
    # 再次正規化處理後的音訊
    enhanced_speech_np = librosa.util.normalize(enhanced_speech_np)
    
    # 保存結果
    sf.write(input_file, enhanced_speech_np, rate)

# In[3]:


# 資料夾路徑
data_folder = r'E:\data_folder'
finished_folder = r'E:\finished_folder'
outputs_folder = r'E:\outputs_folder'
log_folder = r'E:\log_folder'
stage_folder = r'E:\stage_folder'
error_folder = r'E:\error_folder'

log_file = f'{today}log.txt'

# 創建資料夾
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
if not os.path.exists(finished_folder):
    os.makedirs(finished_folder)
if not os.path.exists(outputs_folder):
    os.makedirs(outputs_folder)
if not os.path.exists(log_folder):
    os.makedirs(log_folder)
if not os.path.exists(stage_folder):
    os.makedirs(stage_folder)
if not os.path.exists(error_folder):
    os.makedirs(error_folder)

#Loading models
work_log('import sd model... :( ')
pipeline = Pipeline.from_pretrained("E:\\CS_STT\\config.yaml").to(torch.device("cuda:0"))
work_log('import sd model done :) ')

work_log('import STT model... :( ')

# Plz replace model_path to your real model path
model_path = r'C:\Users\XXXX\.cache\huggingface\hub\faster-whisper-large-v2'
model = WhisperModel(model_path, device="cuda", compute_type="float16")
work_log('import STT model done :)')

# In[4]:


# 定義 logger
def mp3_to_wav(mp3_path):
    command = f'ffmpeg -i {mp3_path} {mp3_path[:-4]}.wav'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    error =''
    output, error = process.communicate()
    return output.decode("utf-8"), error


# In[5]:


def speach_to_text(log_file,audio_path):
    '''ASR主程式'''
    s_time = time.time()
    try:
        df = pd.DataFrame(columns=['start','end','text'])
        segments, info = model.transcribe(audio_path, condition_on_previous_text=False,initial_prompt='電話響鈴',language="zh")
        df = pd.DataFrame([{'start': segment.start, 'end': segment.end, 'text': convert(segment.text)} for segment in segments])
        file_name = os.path.basename(audio_path)
        name, ext = os.path.splitext(file_name)
        df.to_csv(os.path.join(stage_folder, name)+'.tsv', sep='\t', index=False)
        work_log(f'{audio_path} STT Fin, Inference time: {time.time() - s_time}')      
    except Exception as e:
        work_log(f'Error while processing the file: {str(e)}') 
# In[6]:


#speaker_diarization
def sd(audio_path):
    s_time = time.time()
    dia = pipeline(audio_path, num_speakers=2)
    work_log(f'{audio_path} SD fin, Inference time: {time.time() - s_time}')
    file_name = os.path.basename(audio_path)
    name, ext = os.path.splitext(file_name)
    tsv_path = os.path.join(stage_folder, name) + '.sd.tsv'

    with open(tsv_path, 'w') as f:
        f.write('start\tend\tspeaker\n')
        for speech_turn, track, speaker in dia.itertracks(yield_label=True):
            f.write(f"{round((speech_turn.start),2)}\t{round((speech_turn.end),2)}\t{speaker}\n")

    work_log(f'{audio_path} SD fin')


def merge_sd_stt_outputs(audio_path):
    file_name = os.path.basename(audio_path)
    name, ext = os.path.splitext(file_name)
    sd_stage_path = name + '.sd.tsv'
    stt_stage_path = name + '.tsv'

    df_sd = pd.read_csv(os.path.join(stage_folder, sd_stage_path), sep='\t')
    df_stt = pd.read_csv(os.path.join(stage_folder, stt_stage_path), sep='\t')

    # 辨識何者為客戶 (以話少者貼為客戶)
    total_time = df_sd.groupby('speaker')['end'].sum() - df_sd.groupby('speaker')['start'].sum()
    min_speaker = total_time.idxmin()
    df_sd_c = df_sd[df_sd['speaker'] == min_speaker]
    df_sd_c = df_sd_c[(df_sd_c['end'] - df_sd_c['start']) >= 1]

    #改時間軸Error1(start>=end，回找上一行的end)
    for index, row in df_stt.iterrows():
        if index == 0:
            continue
        if row['start'] >= row['end']:
            row['start'] = df_stt.loc[index - 1, 'end']
            df_stt.loc[index, 'start'] = df_stt.loc[index -1, 'end']

    # 合併STT與SD  (目前模糊度tolerance 1.5比較好)
    df_stt = df_stt.sort_values(['start'])
    df_sd_c = df_sd_c.sort_values(['start'])
    df_stt_m1 = pd.merge_asof(df_stt, df_sd_c, on='start', direction='nearest', tolerance=1.5)
    df_stt_m1.loc[
        df_stt_m1.duplicated(subset=['end_y', 'speaker'], keep='first'), ['end_y', 'speaker']] = np.nan  # 重複往前取
    df_stt = df_stt.sort_values(['end'])
    df_sd_c = df_sd_c.sort_values(['end'])
    df_stt_m2 = pd.merge_asof(df_stt, df_sd_c, on='end', direction='nearest', tolerance=1.5)
    df_stt_m2.loc[
        df_stt_m2.duplicated(subset=['start_y', 'speaker'], keep='last'), ['start_y', 'speaker']] = np.nan  # 重複往後取
    concat_df = pd.concat([df_stt, df_stt_m1['speaker'].astype(str), df_stt_m2['speaker'].astype(str)], axis=1,
                          ignore_index=True)
    concat_df = concat_df.set_axis(['start', 'end', 'text', 'sd_start', 'sd_end'], axis=1)

    # 語者回貼 (邏輯為s+s->s, s+null持續到null+s->s, null+s->s)
    concat_df['speaker_c'] = 0
    conti_index = 0
    max_conti_index = 5  # 卡上限 5目前比較好
    for i, row in concat_df.iterrows():
        if row['sd_start'].startswith('SPEAKER'):
            concat_df.at[i, 'speaker_c'] = 1
            if row['sd_end'].startswith('SPEAKER'):
                conti_index = 0
            elif row['sd_end'].startswith('SPEAKER') == False:
                conti_index = 1
        elif row['sd_end'].startswith('SPEAKER'):
            concat_df.at[i, 'speaker_c'] = 1
            conti_index = 0
        elif row['sd_start'].startswith('SPEAKER') == False:
            if conti_index < max_conti_index:
                if conti_index >= 1:
                    concat_df.at[i, 'speaker_c'] = 1
                    conti_index += 1
                elif conti_index == 0:
                    concat_df.at[i, 'speaker_c'] = 0
            else:
                concat_df.at[i, 'speaker_c'] = 0
                conti_index = 0

    # 將判斷標籤置換成語者標籤
    concat_df['speaker_c'].replace({0: 'A', 1: 'C'}, inplace=True)
    concat_df.drop(['sd_start', 'sd_end'], axis=1, inplace=True)
    # 存取結果
    output_path = os.path.join(outputs_folder, name) + '.itg.csv'
    concat_df.to_csv(output_path, index=False, encoding='big5',errors='ignore') # 留存dataframe版本
    # 移動原STT結果到outputs資料夾
    for ext in ['.tsv']:
        text_path = os.path.join(stage_folder, name + ext)
        shutil.move(text_path, os.path.join(outputs_folder, name + ext))

    return concat_df
# In[10]:
class AudioHandler(FileSystemEventHandler):
    def on_created(self, event):
        time.sleep(3)
        run()

def run():
    audio_files=[]
    # 讀取轉換檔案清單
    audio_files = glob(os.path.join(data_folder , '*.wav')) + \
                  glob(os.path.join(data_folder , '*.mp3'))
    # print(f'本次轉換清單: {audio_files}')
    work_log(f'本次轉換清單 {audio_files}')

    # STT
    for audio_path in audio_files:
        try:    
            # 統一轉 wav 來做
            if audio_path[-3:] == 'mp3' or audio_path[-3:] == 'MP3':
                try:
                    if ' ' in audio_path:
                        audio_path_n=''
                        # Create a new file name without spaces
                        audio_path_n = audio_path.replace(' ', '_')
                        # Rename the file
                        os.rename(audio_path, audio_path_n)
                        work_log(f'Renamed "{audio_path}" to "{audio_path_n}"')
                        _,e = mp3_to_wav(audio_path_n)
                        work_log(f'{audio_path_n} to wav 轉換完成')
                        shutil.move(audio_path_n, os.path.join(finished_folder, audio_path_n.split('\\')[-1]))
                        audio_path = audio_path_n[:-4] + '.wav'
                    else:
                        _,e = mp3_to_wav(audio_path)
                        work_log(f'{audio_path} to wav 轉換完成')
                        shutil.move(audio_path, os.path.join(finished_folder, audio_path.split('\\')[-1]))
                        audio_path = audio_path[:-4] + '.wav'
                except:
                    work_log(f'{audio_path} to wav 轉換失敗:{e}')
                    shutil.move(audio_path, os.path.join(error_folder, audio_path.split('\\')[-1]))
                
            try:
                denoise(audio_path)
                work_log(log_file, f'{audio_path} 降噪完成')
            except:
                pass
            speach_to_text(model,audio_path)
            #幻覺重跑
            i = 1
            while True:
                file_name = os.path.basename(audio_path)
                name, ext = os.path.splitext(file_name)
                sd_stage_path = name + '.sd.tsv'
                stt_stage_path = name + '.tsv'
                df_stt = pd.read_csv(os.path.join(stage_folder, stt_stage_path), sep='\t')
                df_stt=df_stt.dropna()
                df_stt = df_stt[(df_stt['end'] - df_stt['start']) >= 0.5]
                lastrows = df_stt['text'].tail()
                if not lastrows.astype(str).nunique() == 1:
                    break
                elif i == 3:
                    work_log(f'------------{name} STT Failed :((-------------')
                    break
                else:
                    work_log(f'{name} STT Again {i}):((')
                    speach_to_text(model,audio_path)
                    i += 1                  
            #語者分離
            if i ==3:
                shutil.move(audio_path, os.path.join(error_folder , audio_path.split('\\')[-1]))
            else:
                sd(audio_path)
                merge_sd_stt_outputs(audio_path)
                # 移動到 finished 資料夾
                shutil.move(audio_path, os.path.join(finished_folder , audio_path.split('\\')[-1]))
            # 錯字修正
                db = cx_Oracle.connect('username/password@ip:port/DB_Name',encoding='UTF-8', nencoding='UTF-8')
                cur = db.cursor()
                file = os.path.join(outputs_folder,audio_path.split('\\')[-1][:-4]+'.itg.csv')
                csv_df = pd.read_csv(file, encoding='big5')
                csv_df['text'] = csv_df['text'].astype(str)
                for index, row in csv_df.iterrows():
                    csv_df.at[index,'text'] = replace_text(row['text'])
                    pattern = r"([一二三四五六七八九十百千]+)"
                    matches = re.findall(pattern, csv_df.at[index,'text'])
                    converted = []
                    for match in matches:
                        try:
                            if len(match)>1:
                                number = cn2an.cn2an(match)
                                converted.append(number)
                            else:
                                converted.append(match)
                        except:
                            converted.append(match)
                    for i in range(len(matches)):
                        csv_df.at[index,'text'] = csv_df.at[index,'text'].replace(matches[i],str(converted[i]),1)
                csv_file = name+'.itg.csv'
                csv_df.to_csv(os.path.join(outputs_folder,csv_file) ,index = False ,encoding='big5',errors='ignore')
                db.close()
                txt_market_path = os.path.join(outputs_folder,audio_path.split('\\')[-1][:-4]+'.itg.txt')
                with open(txt_market_path, 'w', encoding='utf-8') as file:
                    for index, row in csv_df.iterrows():
                        speaker_prefix, text = row['speaker_c'], row['text']
                        file.write(f"{speaker_prefix}: {text}\n")
        except Exception as e:
            work_log(f'{audio_path} stt fail.')
            work_log(f'Error: {str(e)}')
            shutil.move(audio_path, os.path.join(finished_folder , audio_path.split('\\')[-1]))


if __name__ == '__main__':
    work_log('FB Holding STT Service Start.')
    # initial check
    run()

    # setting handler
    event_handler = AudioHandler()
    observer = Observer()
    observer.schedule(event_handler, path=data_folder , recursive=True)
    observer.start()

    try:
        while observer.is_alive():
            observer.join(1)
    finally:
        #請除記憶體占用
        del model
        del pipeline
        gc.collect()
        torch.cuda.empty_cache()
        observer.stop()
        observer.join()
