python run_crypteagle.py --dataset darknet2020_block
python run_crypteagle.py --dataset unsw_nb15
python run_crypteagle.py --dataset cic_ids2017

cd CryptEAGLE-M
python server.py
python client.py --dataset darknet2020_block --data_dir /root/project/reon/processed_data --cid 0
python client.py --dataset darknet2020_block --data_dir /root/project/reon/processed_data --cid 1