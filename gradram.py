def get_grad_cam(model,image_pil):
	normalize =  transforms.Normalize((0.5,), (0.5,))
	preprocess = transforms.Compose([
   	transforms.Resize((224,224)),
   	transforms.ToTensor(),
   	normalize
	])
	img_tensor = preprocess(img_pil)
	img_tensor = img_tensor.to(device)
	img_tensor=img_tensor.unsqueeze(0)
	mem_score = model(img_tensor)
	size_upsample = (224, 224)
	bz, nc, h, w = feature_blobs[-1].shape
	cam = np.dot(weight_softmax.reshape(1,-1),feature_blobs[-1].reshape((nc, h*w)))
	cam = cam.reshape(h, w)
	cam = cam - np.min(cam)
	cam_img = cam / np.max(cam)
	cam_img = np.uint8(255 * cam_img)
	up_sampled_image =cv2.resize(cam_img, size_upsample)
	img = np.array(image_pil)
	height, width, _ = img.shape
	heatmap = cv2.applyColorMap(cv2.resize(up_sampled_image,(width, height)), cv2.COLORMAP_JET)
	result = heatmap * 0.3 + img * 0.5
	cv2.imwrite('CAM.jpg', result)
	return result