import requests
from bs4 import BeautifulSoup
import os
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from matplotlib import pyplot as plt
import openai  # If OpenAI API is used for some task

def preprocess_image(image_path):
    """Preprocess the image (convert to grayscale, resize, denoise)."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Error reading image: {image_path}")
        return None
    img = cv2.resize(img, (256, 256))
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Denoise
    return img

def compare_images(image1, image2):
    """Compare two images using SSIM (Structural Similarity Index)."""
    img1 = preprocess_image(image1)
    img2 = preprocess_image(image2)
    
    if img1 is None or img2 is None:
        return False

    print(f"Comparing {image1} with {image2}")
    similarity, diff = ssim(img1, img2, full=True)
    print(f"SSIM: {similarity}")

    if similarity > 0.3:  # Threshold for similarity (can be adjusted)
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(img1, cmap='gray')
        plt.title('Reference Image')

        plt.subplot(1, 3, 2)
        plt.imshow(img2, cmap='gray')
        plt.title('Downloaded Image')

        plt.subplot(1, 3, 3)
        plt.imshow(diff, cmap='gray')
        plt.title(f'Difference (SSIM: {similarity:.4f})')

        plt.show()
        return True

    return False

def is_relevant_image(image_url):
    """Check if an image is relevant (i.e., not a logo or an icon)."""
    if "logo" in image_url.lower() or "icon" in image_url.lower() or "data:image" in image_url.lower():
        return False
    return True

def download_images(query, folder, max_images=200):
    """Download images based on a Google search query and save them in a folder."""
    search_url = f"https://www.google.com/search?q={query}&tbm=isch"
    headers = {"User-Agent": "Mozilla/5.0"}

    if not os.path.exists(folder):
        os.makedirs(folder)

    image_count = 0
    index = 0

    while image_count < max_images:
        print(f"Fetching page: {search_url}")
        try:
            response = requests.get(search_url, headers=headers)
            response.raise_for_status()  # Raise an error for bad status
        except requests.RequestException as e:
            print(f"Error fetching page: {e}")
            return  # Exit on failure

        soup = BeautifulSoup(response.text, 'html.parser')
        images = soup.find_all("img")

        if not images:
            print("No images found on this page.")
            return

        for img in images:
            if 'src' in img.attrs:
                img_url = img['src']
                if not is_relevant_image(img_url):
                    continue

                try:
                    img_data = requests.get(img_url).content
                    image_path = os.path.join(folder, f"{index}.jpg")
                    with open(image_path, 'wb') as handler:
                        handler.write(img_data)
                    print(f"Downloaded: {image_path}")
                    image_count += 1
                    index += 1
                    yield image_path
                except Exception as e:
                    print(f"Failed to download {img_url}: {e}")

            if image_count >= max_images:
                break

        # Update to next page logic (if applicable)
        next_button = soup.select_one('a#pnnext')
        if not next_button:
            print("No next page button found.")
            return
        search_url = 'https://www.google.com' + next_button['href']

def list_files_in_folder(folder):
    """List all files in a folder."""
    files = os.listdir(folder)
    print(f"Files in {folder}: {files}")

def main(reference_image, vehicle_data=None):
    """Main function to download images, compare them with reference image, and display results."""
    # Step 1: Validate vehicle data
    if not vehicle_data or 'vehicle_id' not in vehicle_data:
        raise ValueError("Invalid or missing vehicle data.")

    # Step 2: Ensure reference image exists
    if not os.path.exists(reference_image):
        raise FileNotFoundError(f"Reference image not found: {reference_image}")
    else:
        print(f"Reference image found: {reference_image}")

    query = "car+plate+number+in+street+in+jordan"
    folder = "C:\\Users\\Dell\\Desktop\\Car_Number_Plate_Detection-master\\downloaded_images"
    downloaded_images = list(download_images(query, folder, max_images=200))  # Increased max_images

    # Step 3: Ensure images were downloaded
    if not downloaded_images:
        raise Exception("No images downloaded from Google.")

    # Step 4: Compare reference image with each downloaded image
    match_found = False
    for downloaded_image in downloaded_images:
        try:
            similarity_score = compare_images(reference_image, downloaded_image)
            if similarity_score:
                print(f"Match found: {downloaded_image}")
                cv2.imshow("Matching Image", cv2.imread(downloaded_image))
                cv2.waitKey(0)  # Display the matched image
                cv2.destroyAllWindows()
                match_found = True
                break
        except Exception as e:
            print(f"Error comparing images: {e}")

    if not match_found:
        print("No match found.")
    
    # List files in the folder to check the downloaded images
    list_files_in_folder(folder)

if __name__ == "__main__":
    reference_image = 'processed_image.jpg'  # Replace with your MATLAB-generated reference image

    # Example of vehicle data input (replace with actual data as needed)
    vehicle_data = {
        'vehicle_id': 123,
        'license_plate': 'XYZ 1234'
    }

    try:
        main(reference_image, vehicle_data)
    except ValueError as e:
        print(f"Error: {e}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
