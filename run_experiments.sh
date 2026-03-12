python run_crypteagle.py --dataset darknet2020_block
python run_crypteagle.py --dataset unsw_nb15
python run_crypteagle.py --dataset cic_ids2017

python server.py
cd CryptEAGLE-M
python client.py --dataset cic_ids2017 --data_dir /root/project/reon/processed_data --cid 0
cd CryptEAGLE-M
python client.py --dataset cic_ids2017 --data_dir /root/project/reon/processed_data --cid 1