# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 14:37:03 2025

@author: HarveyC
"""

import tkinter as tk
from tkinter import ttk
import cx_Oracle
import pandas as pd
from datetime import datetime
import base64
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5 as PKCS1_cipher
import os 


##DB login
def get_key(key_file):
    with open(key_file) as f:
        data = f.read()
        key = RSA.importKey(data)
    return key

def decrypt_data(encrypt_msg):
    private_key = get_key(r'E:\QAs\rsa_private_key.pem')
    cipher = PKCS1_cipher.new(private_key)
    back_text = cipher.decrypt(base64.b64decode(encrypt_msg), 0)
    return back_text.decode('utf-8')

encrypt_msg='RSA加密字串'
""" 從 Oracle DB 取得資料，並根據篩選條件過濾 """
login = decrypt_data(encrypt_msg)
conn = cx_Oracle.connect(login,encoding='UTF-8', nencoding='UTF-8')

# 資料庫連線設定
def get_data(start_date='', end_date='', agt_id='', product_type=''):
    """ 從 Oracle DB 取得資料，並根據篩選條件過濾 """
    login = decrypt_data(encrypt_msg)
    conn = cx_Oracle.connect(login,encoding='UTF-8', nencoding='UTF-8')
    query = """select to_char(f.create_date,'YYYYMMDD') SEARCH_DATE, m.Agent_Id, m.prod_type, m.product_code,f.FILE_NAME, 'D:/QA/finished'||f.FILE_PATH
    from ims.stt_chk_lst_m m
    left join ims.stt_chk_lst_file f
    on m.stt_lst_seq = f.stt_lst_seq
    where stt_flag = 'STT' """
    params = []

    if start_date:
        query += f" AND to_char(f.create_date,'YYYYMMDD') >= {start_date}"
    if end_date:
        query += f" AND to_char(f.create_date,'YYYYMMDD') <= {end_date}"    
    if agt_id:
        query += f" AND m.Agent_Id = '{agt_id}'"
    if product_type and product_type != "全部":
        query += f" AND m.prod_type = '{product_type}'"
    order = " and file_name is not null ORDER BY m.STT_LST_SEQ, CAST(SEQ AS DECIMAL) ASC"
    query += order
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def update_table():
    """ 更新顯示表格的資料 """
    for row in tree.get_children():
        tree.delete(row)

    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    agt_id = agt_id_entry.get()
    product_type = product_type_var.get()

    data = get_data(start_date, end_date, agt_id, product_type)
    for index, row in data.iterrows():
        tree.insert('', 'end', values=row.tolist())

def on_treeview_click(event):
    """ 處理 Treeview 點擊事件 """
    item = tree.identify('item', event.x, event.y)
    col = tree.identify_column(event.x)
    if col == '#6':  # 檔案路徑欄位
        file_path = tree.item(item, 'values')[5]
        if os.path.exists(file_path):
            os.startfile(file_path)
        else:
            tk.messagebox.showerror("錯誤", f"檔案路徑不存在: {file_path}")

# 建立 GUI
root = tk.Tk()
root.title("Oracle 資料篩選顯示")
root.grid_rowconfigure(1, weight=1)
root.grid_columnconfigure(0, weight=1)

# 篩選條件區塊
tk.Label(root, text="開始日期 (YYYYMMDD):").grid(row=0, column=0)
start_date_entry = tk.Entry(root)
start_date_entry.grid(row=0, column=1)

tk.Label(root, text="結束日期 (YYYYMMDD):").grid(row=0, column=2)
end_date_entry = tk.Entry(root)
end_date_entry.grid(row=0, column=3)

tk.Label(root, text="AGENT_ID:").grid(row=0, column=4)
agt_id_entry = tk.Entry(root)
agt_id_entry.grid(row=0, column=5)

tk.Label(root, text="LF/HP:").grid(row=0, column=6)
product_type_var = tk.StringVar()
product_type_dropdown = ttk.Combobox(root, textvariable=product_type_var, values=["全部", "LF", "HP"])
product_type_dropdown.grid(row=0, column=7)

# 更新按鈕
update_btn = tk.Button(root, text="更新", command=update_table)
update_btn.grid(row=0, column=8)

# 資料表格和滾動條
tree_frame = tk.Frame(root)
tree_frame.grid(row=1, column=0, columnspan=9, sticky='nsew')

tree_scroll_y = tk.Scrollbar(tree_frame, orient="vertical")
tree_scroll_y.pack(side="right", fill="y")

tree = ttk.Treeview(tree_frame, columns=("日期", "AGENT_ID", "產壽險", "商品名稱", "檔案名稱", "檔案路徑"), show="headings",
                    yscrollcommand=tree_scroll_y.set)
tree.heading("日期", text="日期")
tree.heading("AGENT_ID", text="AGENT_ID")
tree.heading("產壽險", text="產壽險")
tree.heading("商品名稱", text="商品名稱")
tree.heading("檔案名稱", text="檔案名稱")
tree.heading("檔案路徑", text="檔案路徑")
tree.pack(side="left", fill="both", expand=True)

# 設定每個欄位的寬度為自動調整
for col in tree["columns"]:
    tree.column(col, width=100, minwidth=100, stretch=True)

tree.pack(side="left", fill="both", expand=True)

tree_scroll_y.config(command=tree.yview)

# 綁定點擊事件
tree.bind("<ButtonRelease-1>", on_treeview_click)
update_table()
# 啟動應用程式
root.mainloop()