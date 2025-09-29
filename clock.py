import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def load_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Failed to load image.")
        return None
    return image

def gamma_correction(image, gamma):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv2.LUT(image, table)

# def Manual_equalizeHist(image):
#     M,N = image.shape
#     total_pixels = M * N
#     L = 256
#     histogram = cv2.calcHist([image], [0], None, [256], [0, 256]).flatten()
    
#     probability = np.zeros(L, dtype=np.float32)
#     for i in range(L):
#         probability[i] = histogram[i] / total_pixels
        
#     cum_sum_array = np.zeros(L, dtype=np.float32)
#     trans_func = np.zeros(L, dtype=np.float32)
#     for i in range(L):
#         cum_sum = 0.0
#         for j in range(i+1):
#             cum_sum += probability[j]
#         cum_sum_array[i] = cum_sum
#         trans_func[i] = (L-1) * cum_sum_array[i]
#         if trans_func[i] > 255:
#             trans_func[i] = 255
            
#     equalized_img = cv2.LUT(image, trans_func.astype(np.uint8))
#     return histogram, probability, cum_sum_array, trans_func, equalized_img

def clahe_equalize(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_eq = cv2.merge((cl, a, b))
    equalized_img = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    gray_eq = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2GRAY)
    return gray_eq

def plot_histogram(img, title, pos):
    plt.subplot(2, 3, pos)
    plt.hist(img.ravel(), bins=256, range=[0, 256], color='black')
    plt.title(title)
    plt.xlabel('Pixel value')
    plt.ylabel('Frequency')
    
def smoothing(image):
    smoothed = cv2.medianBlur(image, 5) 
    smoothed = cv2.bilateralFilter(smoothed, 100, 2, 2)
    smoothed = cv2.GaussianBlur(smoothed, (5,5), 0)
    return smoothed

def increase_sharpness(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharp_image = cv2.filter2D(image, -1, kernel)
    return sharp_image

def gaussianFunction(x,y,sigma):
    return (1/(2*np.pi*sigma**2))*np.exp(-(x**2 + y**2) /(2*sigma**2))

def gaussianKernel_x(size, sigma):
    k = size // 2
    coords = np.arange(-k, k + 1)
    x,y = np.meshgrid(coords, coords)
    gaussval = gaussianFunction(x, y, sigma)
    kernel = -(x / sigma**2) * gaussval
    kernel /= np.sum(np.abs(kernel))
    return kernel

def gaussianKernel_y(size, sigma):
    k = size // 2
    coords = np.arange(-k, k + 1)
    x,y = np.meshgrid(coords, coords)
    gaussval = gaussianFunction(x, y, sigma)
    kernel = -(y / sigma**2) * gaussval
    kernel /= np.sum(np.abs(kernel))
    return kernel

def non_maximum_suppression(grad_mag, angle):
    M, N = grad_mag.shape
    Z = np.zeros((M,N), dtype=np.uint8)
    angle = angle * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255
            
            # Angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = grad_mag[i, j+1]
                r = grad_mag[i, j-1]
            # Angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = grad_mag[i-1, j+1]
                r = grad_mag[i+1, j-1]
            # Angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = grad_mag[i-1, j]
                r = grad_mag[i+1, j]
            # Angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = grad_mag[i-1, j-1]
                r = grad_mag[i+1, j+1]

            if (grad_mag[i,j] >= q) and (grad_mag[i,j] >= r):
                Z[i,j] = grad_mag[i,j]
            else:
                Z[i,j] = 0

    return Z

def double_threshold(image, low_thresh, high_thresh, mid_val):
    output = np.zeros_like(image, dtype=np.uint8)
    output[image >= high_thresh] = 255
    output[image < low_thresh] = 0
    mask = (image >= low_thresh) & (image < high_thresh)
    output[mask] = mid_val
    return output


def hysteresis_thresholding(image, low_thresh, high_thresh, strong_val, weak_val):
    strong_pixel_val = strong_val
    weak_pixel_val = weak_val
    output = np.zeros_like(image, dtype=np.uint8)
    strong_pixels = image >= high_thresh
    weak_pixels = (image < high_thresh) & (image >= low_thresh)
    output[strong_pixels] = strong_pixel_val
    output[weak_pixels] = weak_pixel_val
    pad_output = np.pad(output, 1, mode='constant', constant_values=0)
    rows, cols = image.shape
    for _ in range(20):
        for r in range(rows):
            for c in range(cols):
                pr, pc = r+1, c+1
                if pad_output[pr, pc] == weak_pixel_val:
                    if np.max(pad_output[pr-1:pr+2, pc-1:pc+2]) == strong_pixel_val:
                        pad_output[pr, pc] = strong_pixel_val
    output = pad_output[1:-1, 1:-1]
    output[output == weak_pixel_val] = 0
    return output

def skeletonize(image):
    size = np.size(image)
    skel = np.zeros(image.shape, np.uint8)

    ret, img = cv2.threshold(image, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while not done:
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


image_path = 'input images/input5.jpg'
image = load_image(image_path)
if image is None:
    exit()

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gamma = 1.5
gamma_corrected = gamma_correction(gray, gamma)
gamma_corrected_bgr = cv2.cvtColor(gamma_corrected, cv2.COLOR_GRAY2BGR)

# hist, prob, cdf, trans, eq_img = Manual_equalizeHist(gray)

equalized_img = clahe_equalize(gamma_corrected_bgr)

smoothed = smoothing(equalized_img)
sharp = increase_sharpness(smoothed)

sigma = 1
size = 5
kernel_x = gaussianKernel_x(size, sigma)
kernel_y = gaussianKernel_y(size, sigma)

convo_x = cv2.filter2D(sharp, cv2.CV_32F, kernel_x)
convo_y = cv2.filter2D(sharp, cv2.CV_32F, kernel_y)

norm_x = cv2.normalize(convo_x, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
norm_y = cv2.normalize(convo_y, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# grad_mag = np.sqrt(convo_x**2 + convo_y**2)
# grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

gradient_mag = cv2.magnitude(convo_x.astype(np.float32), convo_y.astype(np.float32) )
grad_norm = np.round(cv2.normalize(gradient_mag, None, 0, 255, cv2.NORM_MINMAX)).astype(np.uint8)

# angle = np.arctan2(convo_y, convo_x) * 180 / np.pi
grad_angle = np.arctan2(convo_y, convo_x)

nms_img = non_maximum_suppression(grad_norm, grad_angle)

low_threshold = 30
high_threshold = 80
mid_value = 128

dt_img = double_threshold(nms_img, low_threshold, high_threshold, mid_value)
final_edges = hysteresis_thresholding(dt_img, low_threshold, high_threshold, 255, 75)  
skeleton = skeletonize(final_edges)


plt.figure(figsize=(12, 18))
plt.subplot(5, 4, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(5, 4, 2)
plt.imshow(gray, cmap='gray')
plt.title('Gray Image')
plt.axis('off')

plt.subplot(5, 4, 3)
plt.imshow(gamma_corrected_bgr, cmap='gray')
plt.title('Gamma Corrected Image')
plt.axis('off')

plt.subplot(5, 4, 4)
plt.imshow(equalized_img, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# plot_histogram(gray, 'Original Histogram', 5)
# plot_histogram(equalized_img, 'Equalized Histogram', 6)

plt.subplot(5, 4, 5)
plt.imshow(smoothed, cmap='gray')
plt.title('Smoothed Image')
plt.axis('off')

plt.subplot(5, 4, 6)
plt.imshow(sharp, cmap='gray')
plt.title('Sharpened Image')
plt.axis('off')

plt.subplot(5, 4, 7)
plt.imshow(norm_x, cmap='gray')
plt.title('Normalized X Gradient')
plt.axis('off')

plt.subplot(5, 4, 8)
plt.imshow(norm_y, cmap='gray')
plt.title('Normalized Y Gradient')
plt.axis('off')

plt.subplot(5, 4, 9)
plt.imshow(gradient_mag, cmap='gray')
plt.title('Gradient Magnitude')
plt.axis('off') 

plt.subplot(5, 4, 10)
plt.imshow(nms_img, cmap='gray')
plt.title('Non-Maximum Suppression')
plt.axis('off') 

plt.subplot(5, 4, 11)
plt.imshow(dt_img, cmap='gray')
plt.title('Double Thresholding')
plt.axis('off')

plt.subplot(5, 4, 12)
plt.imshow(final_edges, cmap='gray')
plt.title('Final Edges after Hysteresis')
plt.axis('off')

plt.subplot(5, 4, 13)
plt.imshow(skeleton, cmap='gray')
plt.title('Skeletonized Image')
plt.axis('off')

# plt.tight_layout()
plt.show()

