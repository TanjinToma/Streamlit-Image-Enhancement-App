
import streamlit as st
import numpy as np
#import skimage.io as ios
import matplotlib. pyplot as plt
from PIL import Image,ImageSequence
import base64
import io
from SessionState_import import SessionState
from cane import cane_2d,cane_3d
from VBET_boosting import LDE_2D,pseudo_3D_LDE


def get_image_download_link_2D(img):
	"""Generates a link allowing the PIL image to be downloaded
	in:  PIL image
	out: href string
	"""
	buffered = io.BytesIO()
	img.save(buffered, format="TIFF")
	img_str = base64.b64encode(buffered.getvalue()).decode()
	href = f'<a href="data:file/tif;base64,{img_str}" download="enhance.tif">Download result</a>'
	return href

def get_image_download_link_3D(img):
    buffered = io.BytesIO()
    img[0].save(buffered, format="TIFF", save_all=True, append_images=img[1:])
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/tif;base64,{img_str}" download="enhance_stack.tif">Download result</a>'
    return href



if __name__=="__main__": 

    st.title('Biomedical Image Enhancement App')
    method = st.sidebar.radio("Choose Enhancement Technique", ('CAEFI', 'VBET','Compare'))
    if method == 'CAEFI':    
        lambda_val = float(st.sidebar.text_input('Enter lambda hyperparameter value:', '0.001'))
        #user_input = float(user_input[0])
    else:
        lambda_val = float(st.sidebar.text_input('Enter lambda hyperparameter value:', '0.001'))

        
    uploaded_file = st.file_uploader("Upload an image")
    
    if uploaded_file is not None: 
        count_slices=Image.open(uploaded_file).n_frames
        #st.write(count_slices)
        if count_slices>1:
            list_slices=[]
            
            with Image.open(uploaded_file) as im:
                for slice in ImageSequence.Iterator(im):
                    list_slices.append(np.array(slice))
            
            array_slices=np.zeros([len(list_slices),list_slices[0].shape[0],list_slices[0].shape[1]])
            for i in range(len(list_slices)):
                array_slices[i,:,:] = list_slices[i]
            #st.write(array_slices.shape)
            # display sidebar to show x-y slices along z-stack
            image=array_slices
            indx = st.slider('Change slice', min_value=1, max_value=image.shape[0], step=1,key='org')
            plt.imshow(image[indx-1,:,:],cmap='gray')
            plt.title("2D Slices in Input Stack")
            st.pyplot()
            # display MIP image of the stack
            mip=np.max(image, axis=0)
            plt.imshow(mip,cmap='gray')
            plt.title("MIP of the Input Stack")
            st.pyplot()
            # declare sessionstate
            z,y,x=array_slices.shape
            session_state = SessionState.get(enh_array=np.zeros(image.shape), mip_im=np.zeros([y,x]))
            
            if st.button('Enhance'): 
                if method == 'CAEFI':        
                    # normalize image
                    image=image.astype(np.double)
                    image=image/np.max(image[:])
                     
                # CANE/CAEFI enhancement
                    with st.spinner('Running CAEFI Enhancement...'):
                        enh_im_stack = cane_3d(image, lambda_val)
                        enh_im_stack[enh_im_stack>1]=1.  
                        enh_im_stack[enh_im_stack<0]=0.
                        enh_im_stack=(enh_im_stack*255).astype(np.uint8)
                        mip_enh=np.max(enh_im_stack, axis=0)
                    st.success('Finished!')
                  
                    session_state.enh_array=enh_im_stack
                    session_state.mip_im=mip_enh
                    
                elif method == 'VBET':
                    image=image.astype(np.double)
                    image=image/np.max(image[:])
                    with st.spinner('Running VBET Enhancement...'):
                        boosted_stack=pseudo_3D_LDE(image)
                        enh_im_stack= cane_3d(boosted_stack, lambda_val)
                        enh_im_stack[enh_im_stack>1]=1.  
                        enh_im_stack[enh_im_stack<0]=0.
                        enh_im_stack=(enh_im_stack*255).astype(np.uint8)
                        mip_enh=np.max(enh_im_stack, axis=0)
                    st.success('Finished!')
                    
                    session_state.enh_array=enh_im_stack
                    session_state.mip_im=mip_enh
                else:
                    image=image.astype(np.double)
                    image=image/np.max(image[:])
                     
                # CANE/CAEFI enhancement
                    with st.spinner('Running CAEFI Enhancement...'):
                        enh_im_stack = cane_3d(image, lambda_val)
                        enh_im_stack[enh_im_stack>1]=1.  
                        enh_im_stack[enh_im_stack<0]=0.
                        enh_im_stack=(enh_im_stack*255).astype(np.uint8)
                        mip_enh_caefi=np.max(enh_im_stack, axis=0)
                    st.success('Finished!')
                    with st.spinner('Running VBET Enhancement...'):
                        boosted_stack=pseudo_3D_LDE(image)
                        enh_im_stack= cane_3d(boosted_stack, lambda_val)
                        enh_im_stack[enh_im_stack>1]=1.  
                        enh_im_stack[enh_im_stack<0]=0.
                        enh_im_stack=(enh_im_stack*255).astype(np.uint8)
                        mip_enh_vbet=np.max(enh_im_stack, axis=0)
                    st.success('Finished!')
                
                    fig = plt.figure()
                    fig.add_subplot(121)
                    plt.title("MIP of CAEFI-enhanced stack", fontsize=8)
                    #plt.tick_params(labelsize=6)
                    plt.axis('off')
                    plt.imshow(mip_enh_caefi, cmap='gray')

                    fig.add_subplot(122)
                    plt.title("MIP of VBET-enhanced stack", fontsize=8)
                    #plt.tick_params(labelsize=6)
                    plt.axis('off')
                    plt.imshow(mip_enh_vbet, cmap='gray')
                    st.pyplot()
#                
            if np.sum(session_state.enh_array):
                indh = st.slider('Change slice', min_value=1, max_value=session_state.enh_array.shape[0], step=1,key='enh')
                plt.imshow(session_state.enh_array[indh-1,:,:],cmap='gray')
                plt.title("2D Slices in Enhanced Stack")
                st.pyplot()   
            
            if np.sum(session_state.mip_im):
                plt.imshow(session_state.mip_im,cmap='gray')
                plt.title("MIP of the Enhanced stack")
                st.pyplot()
            
            if np.sum(session_state.enh_array):
                # prepare for save
                save_slices=[]
                for i in range(session_state.enh_array.shape[0]):
                    #slice_obj=Image.fromarray(array_slices[i,:,:].astype(np.uint8))
                    slice_obj=Image.fromarray(session_state.enh_array[i,:,:])
                    save_slices.append(slice_obj)
            
                st.markdown(get_image_download_link_3D(save_slices), unsafe_allow_html=True)

            
            
        else:
            im=Image.open(uploaded_file)
            image=np.asarray(im)
            #image=image/np.max(image)   # normalize
            #st.write(image.shape)
            plt.imshow(image,cmap='gray') 
            plt.title("Input Image")
            st.pyplot()
            
            if st.button("Enhance"):
                if method == 'CAEFI':        
                    # normalize image
                    image=image.astype(np.double)
                    image=image/np.max(image[:])
            
                # CANE/CAEFI enhancement
                    with st.spinner('Running CAEFI Enhancement...'):
                        enh_im= cane_2d(image, lambda_val)
                        enh_im[enh_im>1]=1.; enh_im[enh_im<0]=0.
                        enh_im=(enh_im*255).astype(np.uint8)
                        
                    st.success('Finished!')
                    plt.imshow(enh_im,cmap='gray')
                    plt.title("Enhanced image by CAEFI")
                    st.pyplot()
                    
                    result = Image.fromarray(enh_im) 
                    st.markdown(get_image_download_link_2D(result), unsafe_allow_html=True)
                    
                elif method == 'VBET':
                    # normalize image
                    image=image.astype(np.double)
                    image=image/np.max(image[:])
                    
                    with st.spinner('Running VBET Enhancement...'):
                        VE=LDE_2D(image*255)
                        VE=VE.astype(np.double)
                        VE=VE/255
        
                        boosted_im=np.maximum(VE,image)
        
                        enh_im= cane_2d(boosted_im, lambda_val)
                        enh_im[enh_im>1]=1.; enh_im[enh_im<0]=0.
                        enh_im=(enh_im*255).astype(np.uint8)
                    st.success('Finished!')
                    plt.imshow(enh_im,cmap='gray')
                    plt.title("Enhanced image by VBET")
                    st.pyplot()
                    
                    result = Image.fromarray(enh_im) 
                    st.markdown(get_image_download_link_2D(result), unsafe_allow_html=True)
                    
                else:
                    # normalize image
                    image=image.astype(np.double)
                    image=image/np.max(image[:])
                    
                    with st.spinner('Running CAEFI Enhancement...'):
                        enh_im_caefi= cane_2d(image, lambda_val)
                        enh_im_caefi[enh_im_caefi>1]=1.; enh_im_caefi[enh_im_caefi<0]=0.
                        enh_im_caefi=(enh_im_caefi*255).astype(np.uint8)    
                    st.success('Finished!')
                    
                    with st.spinner('Running VBET Enhancement...'):
                        VE=LDE_2D(image*255)
                        VE=VE.astype(np.double)
                        VE=VE/255
        
                        boosted_im=np.maximum(VE,image)
        
                        enh_im_vbet= cane_2d(boosted_im, lambda_val)
                        enh_im_vbet[enh_im_vbet>1]=1.; enh_im_vbet[enh_im_vbet<0]=0.
                        enh_im_vbet=(enh_im_vbet*255).astype(np.uint8)
                    st.success('Finished!')
                    
                    
                    fig = plt.figure()
                    fig.add_subplot(121)
                    plt.title("CAEFI enhanced image", fontsize=8)
                    #plt.tick_params(labelsize=6)
                    plt.axis('off')
                    plt.imshow(enh_im_caefi, cmap='gray')

                    fig.add_subplot(122)
                    plt.title("VBET enhanced image", fontsize=8)
                    #plt.tick_params(labelsize=6)
                    plt.axis('off')
                    plt.imshow(enh_im_vbet, cmap='gray')
                    st.pyplot()
                    
#                result = Image.fromarray(enh_im) 
#                st.markdown(get_image_download_link_2D(result), unsafe_allow_html=True)
            
            
            
            
            
            
            

            