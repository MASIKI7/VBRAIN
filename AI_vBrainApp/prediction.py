import nibabel as nib
from fastapi import APIRouter, UploadFile, File, HTTPException
import tensorflow as tf
import numpy as np
from pathlib import Path

router = APIRouter()
TEMPORARY_FILES_PATH = "static/temporary_files"
base_model_path = "saved_models"
