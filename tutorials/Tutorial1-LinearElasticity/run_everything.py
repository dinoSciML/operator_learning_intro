import os

os.system('python generate_data.py')
os.system('python compute_coders.py')
os.system('python reduce_data.py')

os.system('python generate_data.py -output_type observable')
os.system('python compute_coders.py -data_dir data/observable/')
os.system('python reduce_data.py -data_dir data/observable/ -output_basis None')

os.system('python training_loop.py')