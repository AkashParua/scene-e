from object_detection import run_inference 


pred = run_inference('frontFar_BLR-2018-03-22_17-39-26_2_frontFar_000006_r.jpg')
print(pred['classes'])