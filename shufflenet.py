import numpy as np
import onnxruntime
sess = onnxruntime.InferenceSession('shufflenet/model.onnx')
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
prediction, = sess.run([label_name], {input_name: np.ones((1,3,224,224), dtype=np.float32)})
print(prediction)
