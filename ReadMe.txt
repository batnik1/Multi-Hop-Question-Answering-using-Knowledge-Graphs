To train embeddings for the model:
!python main.py  --model TuckER --cuda True  --outfile output --valid_steps 1 --dataset MetaQA_half   --num_iterations 5  --batch_size 256  --l3_reg .00001

To train on MetaQA dataset:
# Things to vary-> hops, epochs, kg type
!python main.py  --mode train --nb_epochs 10 --relation_dim 200 --hidden_dim 256 --gpu 0   --freeze 0 --batch_size 64 --validate_every 4 --hops 1 --lr 0.0005 --entdrop 0.1 --reldrop 0.2 --scoredrop 0.2 --decay 1.0  --model ComplEx --patience 10 --ls 0.0 --use_cuda True --kg_type half

TO test on MetaQA dataset:
# Things to vary-> hops, epochs, kg type
!python main.py  --mode test --nb_epochs 10 --relation_dim 200 --hidden_dim 256 --gpu 0   --freeze 0 --batch_size 64 --validate_every 4 --hops 1 --lr 0.0005 --entdrop 0.1 --reldrop 0.2 --scoredrop 0.2 --decay 1.0  --model ComplEx --patience 10 --ls 0.0 --use_cuda True --kg_type half
