$dataset = 
# 'Laptop_Prices_Dataset', '../UJIndoorLoc', '../communities_and_crime', '../cpu_small', 
# 'water_quality', '../ada_agnostic', '../Basketball_c', '../c131',
'dry_bean_dataset', 'waveform_database_generator', '../letter_recognition', '../Credit_c'
# $tf = "False", "True"

foreach ($data in $dataset) {
    # Add-Content -Path "log.txt" -Value "======== dataset: $data ========`n"
    # foreach ($rf in $tf) {
    #     foreach ($bias in $tf) {
            # Add-Content -Path "log.txt" -Value "---- rf $rf - bias $bias ----`n"
            # D:/anaconda3/envs/pytorch/python.exe ./example.py --dataset $data
            D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset $data
            D:/anaconda3/envs/pytorch/python.exe ./ftt_origin.py --dataset $data
    #     }
    # }
    # Add-Content -Path "log.txt" -Value "==============================`n"
}

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset Laptop_Prices_Dataset --pretrain True --checkpoint finetuned/reg_communities_and_crime
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset Laptop_Prices_Dataset --pretrain True --checkpoint finetuned/reg_cpu_small
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset Laptop_Prices_Dataset --pretrain True --checkpoint finetuned/reg_UJIndoorLoc

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../UJIndoorLoc --pretrain True --checkpoint finetuned/reg_Laptop_Prices_Dataset
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../UJIndoorLoc --pretrain True --checkpoint finetuned/reg_communities_and_crime
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../UJIndoorLoc --pretrain True --checkpoint finetuned/reg_cpu_small

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../communities_and_crime --pretrain True --checkpoint finetuned/reg_Laptop_Prices_Dataset
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../communities_and_crime --pretrain True --checkpoint finetuned/reg_cpu_small
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../communities_and_crime --pretrain True --checkpoint finetuned/reg_UJIndoorLoc

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../cpu_small --pretrain True --checkpoint finetuned/reg_Laptop_Prices_Dataset
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../cpu_small --pretrain True --checkpoint finetuned/reg_communities_and_crime
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../cpu_small --pretrain True --checkpoint finetuned/reg_UJIndoorLoc


D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset water_quality --pretrain True --checkpoint finetuned/bin_ada_agnostic
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset water_quality --pretrain True --checkpoint finetuned/bin_Basketball_c
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset water_quality --pretrain True --checkpoint finetuned/bin_c131

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../ada_agnostic --pretrain True --checkpoint finetuned/bin_Basketball_c
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../ada_agnostic --pretrain True --checkpoint finetuned/bin_c131
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../ada_agnostic --pretrain True --checkpoint finetuned/bin_water_quality

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../Basketball_c --pretrain True --checkpoint finetuned/bin_ada_agnostic
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../Basketball_c --pretrain True --checkpoint finetuned/bin_c131
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../Basketball_c --pretrain True --checkpoint finetuned/bin_water_quality

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../c131 --pretrain True --checkpoint finetuned/bin_ada_agnostic
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../c131 --pretrain True --checkpoint finetuned/bin_Basketball_c
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../c131 --pretrain True --checkpoint finetuned/bin_water_quality


D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset dry_bean_dataset --pretrain True --checkpoint finetuned/mul_Credit_c
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset dry_bean_dataset --pretrain True --checkpoint finetuned/mul_letter_recognition
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset dry_bean_dataset --pretrain True --checkpoint finetuned/mul_waveform_database_generator

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset waveform_database_generator --pretrain True --checkpoint finetuned/mul_Credit_c
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset waveform_database_generator --pretrain True --checkpoint finetuned/mul_dry_bean_dataset
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset waveform_database_generator --pretrain True --checkpoint finetuned/mul_letter_recognition

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../letter_recognition --pretrain True --checkpoint finetuned/mul_Credit_c
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../letter_recognition --pretrain True --checkpoint finetuned/mul_dry_bean_dataset
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../letter_recognition --pretrain True --checkpoint finetuned/mul_waveform_database_generator

D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../Credit_c --pretrain True --checkpoint finetuned/mul_dry_bean_dataset
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../Credit_c --pretrain True --checkpoint finetuned/mul_letter_recognition
D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../Credit_c --pretrain True --checkpoint finetuned/mul_waveform_database_generator