import os
import cv2
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
from vehicle_anomaly_detection import detect_anomaly, vehicle_data
from decision_support_system import get_decision
from download_and_compare import download_images_from_google


# Function to compare images using Structural Similarity Index (SSIM)
def compare_images(reference_image, target_image):
    img1 = cv2.imread(reference_image, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(target_image, cv2.IMREAD_GRAYSCALE)

    if img1 is None or img2 is None:
        print(f"Error reading image: {reference_image if img1 is None else target_image}")
        return 0.0

    img1 = cv2.resize(img1, (256, 256))
    img2 = cv2.resize(img2, (256, 256))
    similarity, _ = ssim(img1, img2, full=True)
    return similarity


# Main function
if __name__ == "__main__":
    # Step 1: Anomaly Detection
    if detect_anomaly(vehicle_data):
        print(f"Anomaly Detected for Vehicle {vehicle_data['vehicle_id']}")
        decision = get_decision(vehicle_data)
        print("Suggested Actions:")
        print(decision)

        # Step 2: Download Images
        search_url = "https://www.google.com/search?q=car+plate+number+in+street+in+jordan&tbm=isch"
        download_folder = "downloaded_images"
        downloaded_images = download_images_from_google(search_url, download_folder, max_images=10)

        # Step 3: Compare Images
        reference_image = "CAR1.jpg"  # Replace with your reference image path
        for downloaded_image in downloaded_images:
            similarity_score = compare_images(reference_image, downloaded_image)
            print(f"Comparison score with {downloaded_image}: {similarity_score}")
            if similarity_score > 0.8:  # Threshold for similarity
                print(f"Match found: {downloaded_image}")
                break
        else:
            print("No match found.")

    else:
        print("No anomalies detected.")
        
    # Step 4: Clean Up
    # Remove downloaded images
   # for filename in os.listdir(download_folder):
       # file_path = os.path.join(download_folder, filename)
        #if os.path.isfile(file_path):
         ##   os.remove(file_path)



