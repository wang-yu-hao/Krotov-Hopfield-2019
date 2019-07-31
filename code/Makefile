output/fig4a.pdf: backprop.py hid_output_train_rand.npy hid_output_test.npy label_train_rand.npy label_test.npy
	python backprop.py 1 8 0.4 a
output/fig4b.pdf: backprop.py hid_output_train_rand.npy hid_output_test.npy label_train_rand.npy label_test.npy
	python backprop.py 4.5 8 1.7 b
output/fig4c.pdf: backprop.py hid_output_train_rand.npy hid_output_test.npy label_train_rand.npy label_test.npy
	python backprop.py 10 12 4.6 c
hid_output_train_rand.npy hid_output_test.npy label_train_rand.npy label_test.npy: mnist_all.mat main1.py
	python main1.py
all: output/fig4a.pdf output/fig4b.pdf output/fig4c.pdf



