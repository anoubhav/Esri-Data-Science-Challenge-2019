from darkflow.net.build import TFNet

options = {"model": "cfg/yolo-2c.cfg", "load": "bin/yolo.weights", "threshold": 0.3, "gpu":1.0, "imgdir": "test_data_images/images"}

tfnet = TFNet(options)

tfnet.predict()