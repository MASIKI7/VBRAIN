from django.contrib.auth.hashers import check_password
from django.shortcuts import render, redirect
from .models import Team
import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy import ndimage
import os
from PIL import Image, ImageEnhance
import ants
from scipy.ndimage import zoom, gaussian_filter
import keras
from datetime import datetime
import shutil
import mysql.connector
from django.core.files.storage import FileSystemStorage
from celery import shared_task
from django.http import JsonResponse
import os
import numpy as np
import nibabel as nib
from datetime import datetime
from matplotlib import pyplot as plt
from skimage.transform import resize
import gcz
from tensorflow.keras.models import load_model



def f1(y_true, y_pred):
    return 1


# Constants
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMG_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
TEMPORARY_FILES_PATH = "static/temporary_files"
destination_directory = TEMPORARY_FILES_PATH

os.makedirs("static/files", exist_ok=True)
base_model_path = "saved_models"

URL = "http://127.0.0.1:8000/"

output_dir = TEMPORARY_FILES_PATH

# Create your views here.




# T2 = keras.models.load_model(
#     os.path.join(base_model_path, "t21_1.pkl"), compile=False
# )


try:
    conn = mysql.connector.connect(
        host="localhost",
        user="blueldch_AIVBRAIN",
        password="mm@ww2001AIVBRAIN",
        database="blueldch_AIVBRAIN"
    )
    cursor = conn.cursor(dictionary=True)
except mysql.connector.Error as err:
    print(f"Error: {err}")
    conn.reconnect()
    cursor = conn.cursor(dictionary=True)

print("8888888888888888888888888888 ", cursor)


def get_cursor():
    if not conn.is_connected():
        conn.reconnect()  # Reconnect if the conn was lost
    return conn.cursor(dictionary=True)


def normalize(image):
    eps = 1e-8
    return (image - np.min(image)) / (np.max(image) - np.min(image) + eps)


def resize_slice(slice_2d, target_size=(256, 256)):
    """Resize 2D slice to the target size."""
    width_factor = target_size[0] / slice_2d.shape[0]
    height_factor = target_size[1] / slice_2d.shape[1]
    return ndimage.zoom(slice_2d, (width_factor, height_factor), order=3)


def extract_and_resize_all_slices(volume, target_size=(256, 256)):
    num_slices = volume.shape[-1]
    resized_slices = [resize_slice(volume[:, :, i], target_size)
                      for i in range(num_slices)]
    return resized_slices


def read_nifti_file(filepath: str) -> np.ndarray:
    """Read a NIfTI file using nibabel."""
    scan = nib.load(filepath)
    return scan.get_fdata()


def save_slice_as_png(slice_2d, filename: str, predicted_files_path: str):
    """Save a 2D slice as a PNG image."""
    slice_image = Image.fromarray((slice_2d * 255).astype(np.uint8))
    slice_image = slice_image.convert("L")  # Convert to grayscale
    slice_image = ImageEnhance.Contrast(slice_image).enhance(2.0)
    slice_image = ImageEnhance.Brightness(slice_image).enhance(1.2)
    file_path = os.path.join(predicted_files_path, filename)
    slice_image.save(file_path)
    return file_path


def predict_vasculature_on_slice(slice_2d):
    """Predict vasculature on a single 2D slice."""
    TRAINED_model = keras.models.load_model(
        os.path.join(base_model_path, "Ub2_i.pkl"), compile=False
    )
    normalized_slice = normalize(slice_2d)
    slice_rgb = np.stack([normalized_slice] * 3, axis=-1)  # Convert to RGB
    slice_rgb = np.expand_dims(slice_rgb, axis=0)  # Expand for batch dimension
    predicted_image = TRAINED_model.predict(slice_rgb)[0, :, :, 0]
    return predicted_image


def increase_resolution(image):
    scale_factor = (1.5, 1.5, 1.5)
    high_res_image = zoom(image, scale_factor, order=3)
    return high_res_image


def resize_volume(img):
    desired_depth = 64
    desired_width = 256
    desired_height = 256
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]
    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height
    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height
    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(
        img, (width_factor, height_factor, depth_factor), order=1)
    return img


def resize_volume_mrasag(img):
    desired_depth = 64
    desired_width = 128  # Changed to 128
    desired_height = 128  # Changed to 128
    current_depth = img.shape[-1]
    current_width = img.shape[0]
    current_height = img.shape[1]

    depth = current_depth / desired_depth
    width = current_width / desired_width
    height = current_height / desired_height

    depth_factor = 1 / depth
    width_factor = 1 / width
    height_factor = 1 / height

    img = ndimage.rotate(img, 90, reshape=False)
    img = ndimage.zoom(
        img, (width_factor, height_factor, depth_factor), order=1)
    return img


# def normalize(volume):
#     volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
#     volume = volume.astype("float32")
#     return volume


def adjust_contrast(volume, min_val=0, max_val=255):
    volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume))
    volume = volume * (max_val - min_val) + min_val
    return volume.astype(np.uint8)


def enhance_lesion(mri_iamge, sigma=1.0):
    smoothed_image = gaussian_filter(mri_iamge, sigma=sigma)
    enhanced_image = mri_iamge - smoothed_image
    return enhanced_image

# def load_image(path: str):
#     image = nib.load(path)
#     image = image.get_fdata().astype(np.float32)
#     image = ndimage.median_filter(image, 3)
#     image = resize_volume(image)
#     image = np.expand_dims(image, axis=3)
#     image = normalize(image)
#     return image


def load_imageM(path: str):
    image = nib.load(path)
    image = image.get_fdata().astype(np.float32)
    image = ndimage.median_filter(image, 3)
    # image = increase_resolution(image)
    image = resize_volume(image)
    image = np.expand_dims(
        image, axis=3
    )
    image = normalize(image)
    return image


def load_imageS(path: str):
    image = nib.load(path)
    image = image.get_fdata().astype(np.float32)
    image = ndimage.median_filter(image, 3)
    image = increase_resolution(image)
    image = resize_volume_mrasag(image)
    image = np.expand_dims(image, axis=3)
    image = np.repeat(image, repeats=3, axis=-1)

    image = normalize(image)
    return image


def load_iu(path: str):
    image = ants.image_read(path)
    image = ants.abp_n4(image).numpy()
    # image = increase_resolution(image)
    image = resize_volume(image)
    image = adjust_contrast(image)
    image = np.expand_dims(image, axis=3)
    # image = np.repeat(image, repeats=3, axis=-1)
    image = normalize(image)
    return image


def home(request):
    return render(request, 'index.html')


def login(request):
    return render(request, 'login.html')


def signin(request):
    context = {}
    if request.method == "POST":
        email = request.POST.get('username')
        password = request.POST.get('password')

        # Replace 'your_table_name' with your actual table name
        query = "SELECT * FROM TeamMembers WHERE EMAIL = %s AND Password = %s"

        # Pass email and password as a tuple
        cursor.execute(query, (email, password))

        # Fetch all results
        rows = cursor.fetchall()

        # Check if any rows were returned
        if rows:

            request.session["username"] = email
            return redirect('dashboard')
        else:
            query = "SELECT * FROM Patients WHERE PEMAIL = %s AND PPASSWORD = %s"

            # Pass email and password as a tuple
            cursor.execute(query, (email, password))

            # Fetch all results
            row = cursor.fetchall()

            if row:
                request.session["username"] = email
                return redirect('patientdash')

            else:

                print("No matching records found.")
                context["error"] = "Invalid Username or Password!"
                return render(request, 'login.html', context)
      # Redirect to the dashboard or other page


def dashboard(request):
    context = {}
    username = request.session.get('username')
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()

    print(rows)
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')

    query = "SELECT * FROM TeamMembers"
    cursor.execute(query)

    # Fetch all rows (use dictionary cursor for easy access)
    # Rows will be a list of dictionaries if using a dictionary cursor
    rows = cursor.fetchall()

    # Add rows to context
    context['team_members'] = rows
    return render(request, 'dashboard.html', context)


def patientdash(request):
    context = {}
    username = request.session.get('username')
    query = "SELECT * FROM Patients WHERE PEMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()

    print(rows)
    for row in rows:
        context["EMAIL"] = row.get('PEMAIL', 'Not Found')
        context["NAME"] = row.get('PNAME', 'Not Found')
        context["PHONE"] = row.get('PPHONE', 'Not Found')
    return render(request, 'Patient/patientdash.html', context)


def processing(request):
    print(request.session.get('user_name'))
    return render(request, 'upload.html')


def patientpro(request):
    return render(request, 'Patient/patientupload.html')


def uploads(request):
   

    Swi = keras.models.load_model(
     os.path.join(base_model_path, "model1t1_swid11modeldee2deep.pkl"), compile=False
    )
    if request.method == "POST":
        context = {}
        filename = request.FILES.get('file')
        patientName = request.POST.get('patientName')
        currentTime = datetime.now().strftime("%Y%m%d%H%M%S")
        filenames = str(patientName)+"_"+str(currentTime)+"_"+str(filename)
        username = request.session.get('username')
        request.session['patientName'] = patientName
        print(filenames)
        path = TEMPORARY_FILES_PATH
        if not os.path.exists(path):
            os.mkdir(path)
        destination_file_path = os.path.join(
            destination_directory, os.path.basename(filenames))
        # shutil.copy(filenames, destination_file_path)
        fs = FileSystemStorage(location=destination_directory)
        filenamed = fs.save(filenames, filename)

        # Full path to the saved file
        uploaded_file_url = fs.url(filenamed)
        db_url = URL+"temporary_files/"+filenamed

        query = """
            INSERT INTO Patient (NAME, EMAIL, DATE, UPLOADS)
            VALUES (%s, %s, %s, %s)
         """
        values = (patientName, username, currentTime, db_url)

        saved_file_path = fs.path(filenamed)

        # Load image
        image = load_imageM(saved_file_path)
        imageS = load_imageS(saved_file_path)
        imageS = np.expand_dims(image, axis=0)
        image = np.expand_dims(image, axis=0)
        image2 = load_iu(saved_file_path)
        image2 = np.expand_dims(image2, axis=0)
        print("99999999999999999999999999: ", db_url)
        print("Image shape: ", (image.shape))
        print("ImageS shape: ", (imageS.shape))
        print("image2 shape: ", (image2.shape))

        # Predicting with SWI model
        result_SWI = Swi.predict(image2)
        print("SWI result shape: ", (result_SWI[0].shape))

        output_filenames = [
            f"SWI_processed_{filenames}",
        ]

        output_file_paths = [
            os.path.join(output_dir, filename) for filename in output_filenames
        ]

        nib.save(nib.Nifti1Image(
            result_SWI[0], np.eye(4)), output_file_paths[0])

        print(output_file_paths)

        file_location = saved_file_path  # Temporary file location

        # Read NIfTI file and preprocess slices
        volume = read_nifti_file(file_location)
        resized_slices = extract_and_resize_all_slices(volume)

        predicted_files_path = "static/" + str(patientName) + "/temp"
        predictions = []

        # Use os.makedirs to create all necessary directories in the path
        if not os.path.exists(predicted_files_path):
            os.makedirs(predicted_files_path, exist_ok=True)
        for idx, slice_2d in enumerate(resized_slices):
            predicted_slice = predict_vasculature_on_slice(slice_2d)
            file_path = save_slice_as_png(
                predicted_slice, f"predicted_slice_{idx}.png", predicted_files_path)
            predictions.append(file_path)
        print("message Prediction complete., predicted_slices", predictions)

        try:
            # Execute the query
            cursor.execute(query, values)
            conn.commit()
            context['patientName'] = patientName
            print(f"Patient {patientName} inserted successfully!")
        except mysql.connector.Error as err:
            print(f"Error: {err}")

        finally:
            cursor.close()

        return redirect('predicted')


# Define Celery task for background processing
@shared_task
def process_patient_file_task(saved_file_path, output_dir, patientName):
    # Load and process image
    volume = read_nifti_file(saved_file_path)
    resized_slices = extract_and_resize_all_slices(volume)

    # Batch prediction
    predictions = predict_vasculature_on_batch(resized_slices)

    # Save predictions as PNGs
    predicted_files_path = f"static/{patientName}/temp"
    os.makedirs(predicted_files_path, exist_ok=True)
    saved_files = save_batch_slices_as_png(predictions, predicted_files_path)
    return saved_files

# Main upload view


def patientupload(request):
    if request.method == "POST":
        try:
            # Step 1: File Handling
            filename = request.FILES.get('file')
            patientName = request.POST.get('patientName')
            currentTime = datetime.now().strftime("%Y%m%d%H%M%S")
            filenames = f"{patientName}_{currentTime}_{filename}"
            username = request.session.get('username')
            destination_directory = TEMPORARY_FILES_PATH

            if not os.path.exists(destination_directory):
                os.makedirs(destination_directory, exist_ok=True)

            # Save uploaded file
            fs = FileSystemStorage(location=destination_directory)
            filenamed = fs.save(filenames, filename)

            # Generate file paths
            saved_file_path = fs.path(filenamed)
            db_url = f"{URL}temporary_files/{filenamed}"

            # Step 2: Database Entry
            query = """
                INSERT INTO Patient (NAME, EMAIL, DATE, UPLOADS)
                VALUES (%s, %s, %s, %s)
             """
            values = (patientName, username, currentTime, db_url)
            cursor.execute(query, values)
            conn.commit()

            # Step 3: Background Processing
            output_dir = f"{destination_directory}/processed"
            os.makedirs(output_dir, exist_ok=True)

            # Trigger background task
            process_patient_file_task.delay(
                saved_file_path, output_dir, patientName)

            return JsonResponse({
                'status': 'success',
                'message': f"File uploaded and processing started for {patientName}."
            })

        except Exception as e:
            return JsonResponse({
                'status': 'error',
                'message': f"An error occurred: {str(e)}"
            })


def settings(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    return render(request, 'password.html', context)


def logout(request):
    return redirect('/login')


def profile(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    return render(request, 'profile.html', context)


def predicted(request):
    patientName = request.session.get('patientName', 'unknown')
    patient_folder = f"{patientName}/temp"
    num_slices = 79
    context = {}

    username = request.session.get('username')
    print("999999999999999999999 ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="blueldch_AIVBRAIN",
            password="mm@ww2001AIVBRAIN",
            database="blueldch_AIVBRAIN"
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (username,))
        rows = cursor.fetchall()

        print(rows)
        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')

        predictions = [
            f"static/{patient_folder}/predicted_slice_{i}.png" for i in range(num_slices)]
        context['predictions'] = predictions
        context['folder'] = patient_folder
        print(context['folder'])

        return render(request, 'predicted.html', context)

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        conn.reconnect()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query, (username,))
        rows = cursor.fetchall()

        print(rows)
        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')

        predictions = [
            f"static/{patient_folder}/predicted_slice_{i}.png" for i in range(num_slices)]
        context['predictions'] = predictions
        context['folder'] = patient_folder
        print(context['folder'])

        return render(request, 'predicted.html', context)


def visualize(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    query = "SELECT * FROM Patient"
    cursor.execute(query)

    # Fetch all rows (use dictionary cursor for easy access)
    # Rows will be a list of dictionaries if using a dictionary cursor
    rows = cursor.fetchall()

    # Add rows to context
    context['patients'] = rows
    return render(request, 'visualize.html', context)


def changepassword(request):
    context = {}
    old = request.POST.get('CPassword')
    newP = request.POST.get('NPassword')
    repP = request.POST.get('RPassword')

    print(old, newP, repP)
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()

    if (newP == repP):

        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')
            if old != row.get('PASSWORD', 'Not Found'):
                print("Incorrect Old Password")
                context['unfound'] = "UNFOUND"
                return render(request, 'password.html', context)
            else:
                context['success'] = "SUCCESS"
                print("UPDATE HERE")
                query = "UPDATE TeamMembers SET PASSWORD = %s WHERE EMAIL = %s"
                cursor.execute(query, (newP, username))
                conn.commit()
                cursor.close()
                return render(request, 'password.html', context)

    else:
        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')
        context['unmatch'] = "UNMATCH"
        print("Password not Match")
        return render(request, 'password.html', context)


def predictionResults(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    query = "SELECT * FROM Patient"
    cursor.execute(query)

    # Fetch all rows (use dictionary cursor for easy access)
    # Rows will be a list of dictionaries if using a dictionary cursor
    rows = cursor.fetchall()

    # Add rows to context
    context['patients'] = rows
    return render(request, 'predictions.html', context)


def ploadedFiles(request):
    context = {}
    query = "SELECT * FROM Patient"
    try:
        conn = mysql.connector.connect(
            host="localhost",
            user="blueldch_AIVBRAIN",
            password="mm@ww2001AIVBRAIN",
            database="blueldch_AIVBRAIN"
        )
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Preprocess the data to extract the filename
        processed_rows = []
        for row in rows:
            # Assuming 'UPLOADS' contains the full file path
            if 'UPLOADS' in row and row['UPLOADS']:
                # Extract the filename using os.path.basename
                row['upload_filename'] = os.path.basename(row['UPLOADS'])
            else:
                row['upload_filename'] = None
            processed_rows.append(row)  # Append inside the loop

        # Add processed rows to context
        context['patients'] = processed_rows
        return render(request, 'uploaded.html', context)

    except mysql.connector.Error as err:
        print(f"Error: {err}")
        # Attempt to reconnect
        conn.reconnect()
        cursor = conn.cursor(dictionary=True)
        cursor.execute(query)

        # Fetch all rows
        rows = cursor.fetchall()

        # Preprocess the data to extract the filename
        processed_rows = []
        for row in rows:
            # Assuming 'UPLOADS' contains the full file path
            if 'UPLOADS' in row and row['UPLOADS']:
                # Extract the filename using os.path.basename
                row['upload_filename'] = os.path.basename(row['UPLOADS'])
            else:
                row['upload_filename'] = None
            processed_rows.append(row)  # Append inside the loop

        # Add processed rows to context
        context['patients'] = processed_rows
        return render(request, 'uploaded.html', context)


def team(request):
    context = {}
    query = "SELECT * FROM TeamMembers"
    cursor.execute(query)

    # Fetch all rows (use dictionary cursor for easy access)
    # Rows will be a list of dictionaries if using a dictionary cursor
    rows = cursor.fetchall()

    # Add rows to context
    context['team_members'] = rows
    return render(request, 'team.html', context)


def viewPredictions(request):
    context = {}
    patient_name = request.GET.get('patientName', 'default_value')
    print(patient_name)
    patient_folder = f"{patient_name}/temp"
    num_slices = 79
    context = {}

    predictions = [
        f"static/{patient_folder}/predicted_slice_{i}.png" for i in range(num_slices)]
    context['predictions'] = predictions
    context['folder'] = patient_folder
    print(context['folder'])
    return render(request, 'singlePredicted.html', context)


def visualization(request):
    context = {}
    patient_name = request.GET.get('patientName', 'default_value')
    print(patient_name)
    patient_folder = f"{patient_name}/temp"
    num_slices = 79
    context = {}

    predictions = [
        f"static/{patient_folder}/predicted_slice_{i}.png" for i in range(num_slices)]
    context['predictions'] = predictions
    context['folder'] = patient_folder
    print(context['folder'])
    return render(request, 'viewer.html', context)


def swiResult(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)

    try:
        # Ensure the cursor is connected
        cursor = get_cursor()

        # Fetch user details
        query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"
        cursor.execute(query, (username,))
        rows = cursor.fetchall()
        
        if rows:  # If rows are not empty
            row = rows[0]
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')
        else:
            context["EMAIL"] = context["NAME"] = context["PHONE"] = "Not Found"

        # Fetch all patients
        query = "SELECT * FROM Patient"
        cursor.execute(query)
        context['patients'] = cursor.fetchall()

    except mysql.connector.Error as e:
        print(f"Database error: {e}")
        context["error"] = "Database connection error. Please try again later."
    finally:
        cursor.close()  # Always close the cursor after use

    return render(request, 'swi.html', context)





def pdResult(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    query = "SELECT * FROM Patient"
    cursor.execute(query)

    # Fetch all rows (use dictionary cursor for easy access)
    # Rows will be a list of dictionaries if using a dictionary cursor
    rows = cursor.fetchall()

    # Add rows to context
    context['patients'] = rows
    return render(request, 'pd.html', context)


def t2Result(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    query = "SELECT * FROM Patient"
    cursor.execute(query)

    # Fetch all rows (use dictionary cursor for easy access)
    # Rows will be a list of dictionaries if using a dictionary cursor
    rows = cursor.fetchall()

    # Add rows to context
    context['patients'] = rows
    return render(request, 't2.html', context)


def mraResult(request):
    context = {}
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    query = "SELECT * FROM Patient"
    cursor.execute(query)

    # Fetch all rows (use dictionary cursor for easy access)
    # Rows will be a list of dictionaries if using a dictionary cursor
    rows = cursor.fetchall()

    # Add rows to context
    context['patients'] = rows
    return render(request, 'mra.html', context)


def start_processing(request):
    """Initial view to render the loading page."""
    patient_name = request.GET.get('patientName', 'default_value')
    
    # Pass patient name to the template
    context = {"patient_name": patient_name}
    context['message'] = "Please wait, will redirect you soon..."

    # Render the loading page
    return render(request, 'loading.html', context)


def swiViewPredictions(request):
    context = {}
    patient_name = request.GET.get('patientName', 'default_value')
    print("Patient Name: ", patient_name)

    # Step 1: Locate the uploaded file in the correct folder
    temp_folder = os.path.join('static', 'temporary_files')  # Make sure to reference the correct path
    patient_folder = f"SWI_{patient_name}"
    uploaded_file_path = None

    # Search for the file in the temporary_files folder
    for file_name in os.listdir(temp_folder):
        if file_name.startswith(patient_name) and (file_name.endswith(".nii.gz") or file_name.endswith(".nii")):
            uploaded_file_path = os.path.join(temp_folder, file_name)
            break

    if not uploaded_file_path:
        context["error"] = "File not found for the patient."
        return render(request, 'error.html', context)

    print("Uploaded File Path: ", uploaded_file_path)

    # Check if predictions already exist
    output_folder = os.path.join('static', patient_folder)
    predicted_files_path = output_folder
    predictions_exist = os.path.exists(predicted_files_path) and os.listdir(predicted_files_path)

    if predictions_exist:
        # If predictions already exist, load and render them
        predictions = [os.path.join(predicted_files_path, f) for f in os.listdir(predicted_files_path) if f.endswith('.png')]
        context['predictions'] = predictions
        context['folder'] = patient_folder

        username = request.session.get('username')
        print("Logged in as: ", username)
        query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"
        cursor.execute(query, (username,))
        rows = cursor.fetchall()

        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')

        return render(request, 'swiSinglePredicted.html', context)

    # If predictions do not exist, process and generate them
    os.makedirs(output_folder, exist_ok=True)
    context['message'] = "Processing, please wait..."

    try:
        # Load the SWI model
        Swi = keras.models.load_model(
            os.path.join(base_model_path, 'model1t1_swid11modeldee2deep.pkl'), compile=False
        )

        # Call the function to process the NII file and save slices
        image2 = load_iu(uploaded_file_path)
        image2 = np.expand_dims(image2, axis=0)
        print("image2 shape: ", (image2.shape))

        # Predicting with SWI model
        result_SWI = Swi.predict(image2)
        print("SWI result shape: ", (result_SWI[0].shape))

        # Read NIfTI file and preprocess slices
        volume = read_nifti_file(uploaded_file_path)
        resized_slices = extract_and_resize_all_slices(volume)

        predictions = []
        for idx, slice_2d in enumerate(resized_slices):
            predicted_slice = predict_vasculature_on_slice(slice_2d)
            file_path = save_slice_as_png(
                predicted_slice, f"predicted_slice_{idx}.png", predicted_files_path)
            predictions.append(file_path)
        print("Prediction complete. Predicted slices: ", predictions)

        # Pass prediction images and folder information to context
        context['predictions'] = predictions
        context['folder'] = patient_folder

        # Add patient and user details to the context
        username = request.session.get('username')
        print("Logged in as: ", username)
        query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"
        cursor.execute(query, (username,))
        rows = cursor.fetchall()

        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')

        return render(request, 'swiSinglePredicted.html', context)

    except Exception as e:
        print(f"Error during processing: {e}")
        context["error"] = "An error occurred during processing."
        return render(request, 'error.html', context)

 





def start_processingpd(request):
    """Initial view to render the loading page."""
    patient_name = request.GET.get('patientName', 'default_value')
    
    # Pass patient name to the template
    context = {"patient_name": patient_name}
    context['message'] = "Please wait, will redirect you soon..."

    # Render the loading page
    return render(request, 'loadingpd.html', context)



def pdViewPredictions(request):
    context = {}
    patient_name = request.GET.get('patientName', 'default_value')
    print("Patient Name: ", patient_name)

    # Step 1: Locate the uploaded file in the correct folder
    temp_folder = os.path.join('static', 'temporary_files')  # Make sure to reference the correct path
    patient_folder = f"PD_{patient_name}"
    uploaded_file_path = None

    # Search for the file in the temporary_files folder
    for file_name in os.listdir(temp_folder):
        if file_name.startswith(patient_name) and (file_name.endswith(".nii.gz") or file_name.endswith(".nii")):
            uploaded_file_path = os.path.join(temp_folder, file_name)
            break

    if not uploaded_file_path:
        context["error"] = "File not found for the patient."
        return render(request, 'error.html', context)

    print("Uploaded File Path: ", uploaded_file_path)

    # Check if predictions already exist
    output_folder = os.path.join('static', patient_folder)
    predicted_files_path = output_folder
    predictions_exist = os.path.exists(predicted_files_path) and os.listdir(predicted_files_path)

    if predictions_exist:
        # If predictions already exist, load and render them
        predictions = [os.path.join(predicted_files_path, f) for f in os.listdir(predicted_files_path) if f.endswith('.png')]
        context['predictions'] = predictions
        context['folder'] = patient_folder

        username = request.session.get('username')
        print("Logged in as: ", username)
        query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"
        cursor.execute(query, (username,))
        rows = cursor.fetchall()

        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')

        return render(request, 'pdSinglePredicted.html', context)

    # If predictions do not exist, process and generate them
    os.makedirs(output_folder, exist_ok=True)
    context['message'] = "Processing, please wait..."

    try:
        # Load the SWI model
        
        Pd = load_model(
            os.path.join(base_model_path, 'PD1_1.pkl'), 
            custom_objects={'f1':f1},  # Ignore custom objects
            compile=False       # Avoid recompiling the model
        )

        # Call the function to process the NII file and save slices
        image = load_imageM(uploaded_file_path)
        image = np.expand_dims(image, axis=0)
        print("image shape: ", (image.shape))

        # Predicting with SWI model
        result_PD = Pd.predict(image)
        print("PD result shape: ", (result_PD[0].shape))

        # Read NIfTI file and preprocess slices
        volume = read_nifti_file(uploaded_file_path)
        resized_slices = extract_and_resize_all_slices(volume)

        predictions = []
        for idx, slice_2d in enumerate(resized_slices):
            predicted_slice = predict_vasculature_on_slice(slice_2d)
            file_path = save_slice_as_png(
                predicted_slice, f"predicted_slice_{idx}.png", predicted_files_path)
            predictions.append(file_path)
        print("Prediction complete. Predicted slices: ", predictions)

        # Pass prediction images and folder information to context
        context['predictions'] = predictions
        context['folder'] = patient_folder

        # Add patient and user details to the context
        username = request.session.get('username')
        print("Logged in as: ", username)
        query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"
        cursor.execute(query, (username,))
        rows = cursor.fetchall()

        for row in rows:
            context["EMAIL"] = row.get('EMAIL', 'Not Found')
            context["NAME"] = row.get('NAME', 'Not Found')
            context["PHONE"] = row.get('PHONE', 'Not Found')

        return render(request, 'pdSinglePredicted.html', context)

    except Exception as e:
        print(f"Error during processing: {e}")
        context["error"] = "An error occurred during processing."
        return render(request, 'error.html', context)


def mraViewPredictions(request):
    context = {}
    patient_name = request.GET.get('patientName', 'default_value')
    print(patient_name)
    patient_folder = f"{patient_name}/temp"
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    num_slices = 79
    context = {}

    predictions = [
        f"static/{patient_folder}/predicted_slice_{i}.png" for i in range(num_slices)]
    context['predictions'] = predictions
    context['folder'] = patient_folder
    print(context['folder'])
    return render(request, 'mraSinglePredicted.html', context)


def t2ViewPredictions(request):
    context = {}
    patient_name = request.GET.get('patientName', 'default_value')
    print(patient_name)
    patient_folder = f"{patient_name}/temp"
    username = request.session.get('username')
    print("000000000000000000000000000: ", username)
    query = "SELECT * FROM TeamMembers WHERE EMAIL = %s"

    cursor.execute(query, (username,))
    rows = cursor.fetchall()
    for row in rows:
        context["EMAIL"] = row.get('EMAIL', 'Not Found')
        context["NAME"] = row.get('NAME', 'Not Found')
        context["PHONE"] = row.get('PHONE', 'Not Found')
    num_slices = 79
    context = {}

    predictions = [
        f"static/{patient_folder}/predicted_slice_{i}.png" for i in range(num_slices)]
    context['predictions'] = predictions
    context['folder'] = patient_folder
    print(context['folder'])
    return render(request, 't2SinglePredicted.html', context)




