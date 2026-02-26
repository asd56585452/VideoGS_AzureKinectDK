DATASETS_NAME_1="HiFi4G_Dataset"
DATASETS_NAME_2="4K_Actor1_Greeting"

# 先將路徑抽離成變數，程式碼會更乾淨與好維護
SRC_DIR="./datasets/$DATASETS_NAME_1/$DATASETS_NAME_2"
TARGET_DIR="./datasets/$DATASETS_NAME_1/$DATASETS_NAME_2"  # 若要解壓成雙層改為 TARGET_DIR="./datasets/$DATASETS_NAME_1/$DATASETS_NAME_2"
ZIP_FILE="$TARGET_DIR/${DATASETS_NAME_2}.zip"

# 合併、解壓縮並刪除暫存 zip
cat "$SRC_DIR/${DATASETS_NAME_2}.zip.part"* > "$ZIP_FILE" \
    && unzip -qo "$ZIP_FILE" -d "$TARGET_DIR" \
    && rm "$ZIP_FILE"

# 執行 preprocess 腳本時因為已經 cd，所以需加上 ../ 尋找外層的 datasets
# 並將路徑補上 $DATASETS_NAME_1 讓輸出維持在 HiFi4G_Dataset 目錄下
cd preprocess \
    && python hifi4g_process.py --input "../datasets/$DATASETS_NAME_1/$DATASETS_NAME_2" --output "../datasets/$DATASETS_NAME_1/${DATASETS_NAME_2}_preprocess" --move true \
    && cd .. 
    # && rm -rf "$SRC_DIR"
