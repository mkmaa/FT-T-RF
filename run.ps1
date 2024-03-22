# $dataset = 
# # 'Laptop_Prices_Dataset', '../UJIndoorLoc', '../communities_and_crime', '../cpu_small', 
# # 'water_quality', '../ada_agnostic', '../Basketball_c', '../c131',
# 'dry_bean_dataset', 'Mobile_Price_Classification', '../drug_consumption', '../letter_recognition'
# # $tf = "False", "True"

# foreach ($data in $dataset) {
#     # Add-Content -Path "log.txt" -Value "======== dataset: $data ========`n"
#     # foreach ($rf in $tf) {
#     #     foreach ($bias in $tf) {
#             # Add-Content -Path "log.txt" -Value "---- rf $rf - bias $bias ----`n"
#             # D:/anaconda3/envs/pytorch/python.exe ./example.py --dataset $data
#             D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset $data
#             D:/anaconda3/envs/pytorch/python.exe ./ftt_origin.py --dataset $data
#     #     }
#     # }
#     # Add-Content -Path "log.txt" -Value "==============================`n"
# }

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset Mobile_Price_Classification
D:/anaconda3/envs/pytorch/python.exe ./ftt_origin.py --dataset Mobile_Price_Classification

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset dry_bean_dataset --pretrain True --checkpoint finetuned/mul_Mobile_Price_Classification

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../drug_consumption --pretrain True --checkpoint finetuned/mul_Mobile_Price_Classification

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../letter_recognition --pretrain True --checkpoint finetuned/mul_Mobile_Price_Classification

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset Mobile_Price_Classification --pretrain True --checkpoint finetuned/mul_dry_bean_dataset
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset Mobile_Price_Classification --pretrain True --checkpoint finetuned/mul_letter_recognition
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset Mobile_Price_Classification --pretrain True --checkpoint finetuned/mul_drug_consumption