Step	Script Name         	       |    Input CSV(s)	                             |  Output(s)	
1	    sample_data.py	               |    myntra202305041052.csv	                     |  myntra_sample.csv	
2	    download_images.py      	   |    myntra_sample.csv	                         |  myntra_sample_with_images.csv	
3	    verify_images.py         	   |    myntra_sample_with_images.csv	             |  myntra_sample_clean.csv	
4	    fix_image_filenames.py         |    myntra_sample_clean.csv	                     |  myntra_sample_clean.csv (fixed)	
5	    cnn_feature_extraction.py      |	myntra_sample_clean.csv	                     |  cnn_features.npy, myntra_processed_final.csv	
6	    app_cnn.py	                   |    myntra_processed_final.csv,cnn_features.npy  |  Streamlit app	    


Using the files from the file : HSV_Overview.md
The .npy file is large(>25mb), so couldn't upload it. but you get that running the cnn_feature_extraction.py script