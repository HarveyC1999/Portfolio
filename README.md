# Portfolio
Portfolio ! Include STT, whisper_tuning, GUI .etc

# Local_STT (
## 簡介
###使用需求：本地、windows環境、降噪、STT、語者分離、存入csv檔案
本地語音轉文字STT轉換器，使用地端模型進行語音轉文字轉換，並監聽指定資料夾中的聲音文件自動轉換。

## 功能
- mp3轉wav，for 語者分離
- 使用STT模型(這裡用whisper，本地端)進行語音轉文字轉換
- 監聽指定資料夾中的wav文件
- 自動處理新添加的wav文件並生成對應的文字文件

## 環境
- Windows
- python 3.10
- 依賴包： requirements.txt

# whisper Tuning
## tuning whisper and trun .safetensors into .bin using csv files as datasets

# datasearch_gui
## 連接oracle資料庫，可新增式條件、動態顯示介面，提供file_link到目標資料夾
