# $dataset = 'Laptop_Prices_Dataset', '../communities_and_crime', '../cpu_small', 'water_quality', '../ada_agnostic', '../Basketball_c', '../c131', '../UJIndoorLoc'
$dataset = 'water_quality', '../ada_agnostic', '../Basketball_c', '../c131'
$tf = "False", "True"

foreach ($data in $dataset) {
    # Add-Content -Path "log.txt" -Value "======== dataset: $data ========`n"
    # foreach ($rf in $tf) {
    #     foreach ($bias in $tf) {
            # Add-Content -Path "log.txt" -Value "---- rf $rf - bias $bias ----`n"
            # D:/anaconda3/envs/pytorch/python.exe ./example.py --dataset $data
            D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset $data
    #     }
    # }
    # Add-Content -Path "log.txt" -Value "==============================`n"
}

# D:/anaconda3/envs/pytorch/python.exe ./main.py --dataset ../UJIndoorLoc --pretrain False 
# D:/anaconda3/envs/pytorch/python.exe ./example.py --dataset ../UJIndoorLoc