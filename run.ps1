$dataset = 'CA'
$tf = "False", "True"
# $init = "xavier_uniform", "xavier_normal", "kaiming_uniform", "kaiming_normal"
# $fan = "fan_in", "fan_out"

foreach ($data in $dataset) {
    Add-Content -Path "log.txt" -Value "======== dataset: $data ========`n"
    foreach ($rf in $tf) {
        foreach ($bias in $tf) {
            Add-Content -Path "log.txt" -Value "---- rf $rf - bias $bias ----`n"
            D:/anaconda3/envs/pytorch/python.exe ./package/example.py --dataset $data --rf $rf --bias $bias --init xavier_uniform
            D:/anaconda3/envs/pytorch/python.exe ./package/example.py --dataset $data --rf $rf --bias $bias --init xavier_normal
            D:/anaconda3/envs/pytorch/python.exe ./package/example.py --dataset $data --rf $rf --bias $bias --init kaiming_uniform --fan fan_in
            D:/anaconda3/envs/pytorch/python.exe ./package/example.py --dataset $data --rf $rf --bias $bias --init kaiming_uniform --fan fan_out
            D:/anaconda3/envs/pytorch/python.exe ./package/example.py --dataset $data --rf $rf --bias $bias --init kaiming_normal --fan fan_in
            D:/anaconda3/envs/pytorch/python.exe ./package/example.py --dataset $data --rf $rf --bias $bias --init kaiming_normal --fan fan_out
        }
    }
    Add-Content -Path "log.txt" -Value "==============================`n"
}