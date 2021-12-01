# ReadMe

`./train.sh` 需要传入两个参数，依次为训练数据集、测试数据集，例如

`./train.sh ./AS2_data/train_data ./AS2_data/testdata_raw`



`./test.sh` 需要传入五个参数，依次为输出结果路径、测试数据集和训练好的三个param文件，例如

`./test.sh ./AS2_data/testdata_raw/test.csv ./AS2_data/testdata_raw ./Domain-classifier.param ./Feature-extractor.param ./Label-predictor.param`

