# for KUBRICK
#python3 train_runner.py JNet_529
#python3 finetuning.py   JNet_529
#python3 finetuning.py   JNet_530
#python3 finetuning_with_simulation.py JNet_531

#python3 train_runner.py JNet_533
# for Bengio screen d, GPU=3
python3 train_runner.py JNet_533 --train_align;python3 finetuning.py JNet_533 --train_with_align;python3 train_runner.py JNet_534 --train_align;python3 finetuning.py JNet_534 --train_with_align
# for Bengio screen e, GPU=4
python3 train_runner.py JNet_535 --train_align;python3 finetuning.py JNet_535 --train_with_align
#python3 finetuning_with_simulation.py JNet_536 --train_with_align
# for Bengio screen s, GPU=5
python3 finetuning.py   JNet_538 --train_with_align;python3 train_runner.py JNet_539 --train_align;python3 finetuning.py JNet_539 --train_with_align
# for Bengio screen u, GPU=6
python3 train_runner.py JNet_540 --train_align;python3 finetuning.py JNet_540 --train_with_align
##python3 finetuning_with_simulation.py JNet_541 --train_with_align
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA09_20230915.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA15_20230922.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA15_20230923.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA19_20230923.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA19_20231003.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA23_20231006.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA24_20231009.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA26_20231009.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA28_20231106.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA28_20231109.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA29_20231023.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA29_20231111.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA30_20231030.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA32_20231106.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA32_20231111.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA34_20231110.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA34_20231120.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA37_20231120.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA37_20231124.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA41_20231124.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA41_20231127.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA41_20231209.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA42_20231207.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA44_20231202.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA44_20231204.nd2 JNet_544;
python3 apply.py /home/haruhiko/Downloads/Set_03/MDA44_20231207.nd2 JNet_544;
