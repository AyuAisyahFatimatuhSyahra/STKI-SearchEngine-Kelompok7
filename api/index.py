import os
import sys

# Tambahkan direktori utama ke sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import app

# Handler untuk Vercel Serverless
app = app