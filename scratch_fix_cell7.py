import json

file_path = r'm:\Term 10\Grad\SLR Main\Unified_Dataset_Merger.ipynb'
with open(file_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code' and 'CELL 7: CLEANING & MERGING' in ''.join(cell['source']):
        source_code = [
            "# ============================================================\n",
            "# CELL 7: CLEANING & MERGING THE EXTRACTED DATA\n",
            "# ============================================================\n",
            "\n",
            "print(\"🚀 STARTING THE MERGE & CLEANING PROCESS...\")\n",
            "\n",
            "if 'extracted_data' not in globals() or not extracted_data:\n",
            "    raise ValueError(\"❌ NO DATA FOUND! You must run the cell above, paste your file paths, and click '🚀 Extract & Validate' BEFORE running this cell!\")\n",
            "\n",
            "if isinstance(extracted_data[0], tuple):\n",
            "    raise ValueError(\"❌ This cell is designed for CSV datasets (Letters). NPZ datasets (Words) require NumPy arrays. Please use the original Unified Merger (Cell 1 to 5) at the top of the notebook for Words.\")\n",
            "\n",
            "# 1. Combine all dataframes from the previous step into one giant dataset\n",
            "combined_df = pd.concat(extracted_data, ignore_index=True)\n",
            "\n",
            "# 2. FIX THE 35 CLASSES ISSUE: \n",
            "# Make every label uppercase and strip out accidental spaces so 'a' and ' A ' become 'A'\n",
            "if 'label' in combined_df.columns:\n",
            "    combined_df['label'] = combined_df['label'].astype(str).str.upper().str.strip()\n",
            "\n",
            "# 3. Generate the Final Stats\n",
            "total_samples = len(combined_df)\n",
            "if 'label' in combined_df.columns:\n",
            "    unique_classes = combined_df['label'].unique()\n",
            "    num_classes = len(unique_classes)\n",
            "    \n",
            "    print(f\"\\n✅ Merged {len(extracted_data)} datasets successfully!\")\n",
            "    print(f\"📊 Total Samples: {total_samples:,}\")\n",
            "    print(f\"🏷️ Total Unique Classes: {num_classes}\")\n",
            "    print(f\"🔠 Classes Found: {sorted(unique_classes)}\")\n",
            "    \n",
            "    # 4. Final Warning Check\n",
            "    if num_classes > 29:\n",
            "        print(\"\\n⚠️ WARNING: You still have more than 29 classes. Look at the 'Classes Found' list above.\")\n",
            "        print(\"You might have numbers (0-9) or weird typo labels you need to delete.\")\n",
            "else:\n",
            "    print(f\"\\n✅ Merged {len(extracted_data)} datasets successfully!\")\n",
            "    print(f\"📊 Total Samples: {total_samples:,}\")\n",
            "\n",
            "# 5. Save the final file to your folder\n",
            "output_filename = \"unified_asl_dataset.csv\"\n",
            "combined_df.to_csv(output_filename, index=False)\n",
            "print(f\"\\n💾 Saved clean, combined dataset to: {output_filename}\")\n"
        ]
        cell['source'] = source_code

with open(file_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
