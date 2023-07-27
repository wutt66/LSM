import cv2
import numpy as np
import denoise_normal
import levelset_weighted
# import levelset_notf
# import join_pipe
import post_process
import matplotlib.pyplot as plt
# from skimage.morphology import skeletonize_3d
import time

def process():
    rootpath= r"C:\Users\PETER\PycharmProjects\levelset-ver4.3.2\images\RTheta_img_"
    print (rootpath)
    """DENOISE Parameters"""
    start_range = 0.0
    stop_range = 30.0

    """gaussian params"""
    mean_pipe = 115.69
    std_pipe = 43.79
    mean_bg = 72.22
    std_bg = 27.44

    gauss_param = [[mean_pipe, std_pipe], [mean_bg, std_bg]]
    print (gauss_param)
    """level set"""
    bool_count = True
    iteration = 550

    prob_p = 13797. / 364000
    prob_bg = 350200. / 364000
    eta = 0.25 * np.log10 (prob_bg / prob_p)
    print ("eta: ", eta)
    start_bool = True
    scene = 15  # 15

    for i in range(scene, 145,2): #10, 130
        imagepath = r"C:\Users\PETER\PycharmProjects\levelset-ver4.3.2\images\multilook_0.png"
        img = cv2.imread(imagepath)
        img = img[:, 20:748]  # Crop
        if (start_bool):
            print ("multilook")
            detail = " multilook "
            imagepath = r"C:\Users\PETER\PycharmProjects\levelset-ver4.3.2\images\multilook_0.png"
            img = cv2.imread (imagepath)
            img = img[:, :]  # Crop
        else:
            print ("frame")
            detail = " frame "
            imagepath = rootpath + str (i) + ".jpg"
            img = cv2.imread (imagepath)
            img = img[25:525, :]  # Crop
            start_bool = True
            plt.imshow(img)
            plt.show()
        # """Don't use multilook"""
            imagepath = rootpath + str(i) + ".jpg"
            img = cv2.imread(imagepath)
            img = img[25:525, 20:748]  # Crop

        start = time.time ()
            # print("INPUT SHAPE: ", img.shape)
    """_____IMPORT IMAGE_____"""
    img_rgb = img
    img = cv2.GaussianBlur(img,(9,9),3)
    img = cv2.normalize(img,None,0,255,cv2.NORM_MINMAX)
    #cv2.imshow("img", img)
    """______DENOISE______"""
    img_denoise = denoise_normal.denoise_range(img, start_range, stop_range)
    img_denoise_sh = img_denoise.astype(dtype = np.uint8)
    print("INPUT SHAPE: ", img_denoise_sh.shape)
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\denoise" + str(i) + ".png", img_denoise_sh)
    # cv2.imshow("img_denoise", img_denoise_sh)
    # plt.subplot(121)
    # plt.title("image_histrogram")
    # plt.hist(img_denoise_sh.ravel(), 256, [0, 256]);
    # plt.subplot(122)
    # plt.title("gamma-distribution (red: bg, blue: pipe)")
    # x=np.linspace(0, 256, num=256)
    # plt.plot(x, gamma.pdf(x, shape_bg, loc=loc_bg, scale=scale_bg),'r-', lw = 5, alpha = 0.6, label = 'gamma pdf')
    # # plt.subplot(224)
    # # plt.title("pipeline gamma-distribution")
    # plt.plot(x, gamma.pdf(x, shape_pipe, loc=loc_pipe, scale=scale_pipe), 'b-', lw=5, alpha=0.6, label='gamma pdf')
    # plt.show()
    """Level-set"""
    if (bool_count):
        bool_count = False
        phi_coef = 2.0 #6.0
        init_phi = (-1.0 * phi_coef) * np.ones(img_denoise_sh.shape, 'float32')
        # init_phi[200:400, 50:250] = phi_coef
        # init_phi[200:400, 300:500] = phi_coef
        init_phi[200:450, 450:700] = phi_coef
        print ("FIRST FRAME")
        # rect_pre = np.zeros(img_denoise_sh.shape)
        paired_point_pre = []
        poly_function_pre = 0
        # plt.title("initial phi multilook 10-15")
        # plt.imshow(init_phi)
        # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\231218(multilook&fix_penalty)\frame_10-15_multilook_phi_initial.png")
        # # plt.show()
    else:
        init_phi = phi_added
        #plt.title("phi_added")
        # plt.imshow(phi_added)
        # plt.show()
        print("FRAME ", i)
        iteration = 250
        """suppress"""
        # # eta_mat = 0 * eta_mat
        # # rect_pre = rect_old
        # paired_point_pre = paired_point_now
        # poly_function_pre = poly_function_now
        # plt.imshow(phi_added)
        # plt.title("phi initial "+str(i))
        # # plt.savefig(r"C:\Users\mook\PycharmProjects\LSM\experiment\231218(multilook&fix_penalty)\frame_"+str(i)+"multilook_phi_initial.png")
        # plt.show()

    # phi, joint_point, rect_old = levelset_weighted.levelset_cal(img_denoise, init_phi, iteration, gauss_param, i, eta, rect_pre)

    """tensorflow"""
    print ("Sent  img_denoise:",img_denoise)
    print ("Sent  init_phi:", init_phi)
    print ("Sent  iteration:", iteration)
    print ("Sent  gauss_param:", gauss_param)
    print ("Sent  i:", i)
    print ("Sent  eta:", eta)
    print ("Sent  start_bool:", start_bool)

    phi = levelset_weighted.levelset_cal(img_denoise, init_phi, iteration, gauss_param, i, eta, start_bool)
    end = time.time()
    print("Processing time: %.3f min" % ((end - start) / 60))
    # plt.title("phi")
    # plt.imshow(phi)
    # plt.show()

    ### pos = np.nonzero((0*(phi<=0))+(phi*(phi>0)))
    ### mean_phi_pipe = np.mean(phi[pos])
    # print("mean: ", mean_phi_pipe)

    ### out_thresh = (0 * (phi <= 0)) + (255 * (phi > 0))
    # plt.title("out_thresh")
    # plt.imshow(out_thresh)
    # plt.show()
    # cv2.imwrite(r"C:\Users\mook\PycharmProjects\LSM\experiment\231218(multilook&fix_penalty)\res_multilook.png", out_thresh)
    ### res, canvas, labeled, poly_function=post_process.join(out_thresh, i, poly_function_pre)

    rows, cols = img_denoise_sh.shape
    x = np.arange(0, cols + 1, 76.8)
    y = np.arange(0, rows + 1, 50.0)
    y_new = np.arange(30, 7, -2.0)
    x_new = np.arange(-65, 66, 13.0)

    ### phi_added=((-1.0*phi_coef)*(canvas==0))+(phi_coef*(canvas>0))
    plt.title("result"+detail+str(i))
    ### plt.imshow(phi_added)
    plt.xticks(x, x_new, rotation='vertical')
    plt.yticks(y, y_new)
    plt.xlabel('theta(degree)', fontsize=15)
    plt.ylabel('range(meters)', fontsize=15)
    plt.savefig(r"C:\Users\PETER\PycharmProjects\levelset-ver4.3.2\experiment\frame_" + str(i) + "_result.png")

    ### phi_save = (255*(phi_added>0))+(0*(phi_added<=0))
    ### cv2.imwrite(r"C:\Users\PETER\PycharmProjects\levelset-ver4.3.2\experiment\frame_" + str(i) + "_res.png",phi_save)
    # plt.show()

    ### phi_comp=(0*(phi_added==0))+(255*(phi_added>0))
    ### phi_comp=phi_comp.astype(dtype = np.uint8)
    ### comp=cv2.addWeighted(img_denoise_sh,0.7,phi_comp,0.3,0)
    plt.title("compare result with original image"+detail+str(i))
    ### plt.imshow(comp)
    plt.xticks(x, x_new, rotation='vertical')
    plt.yticks(y, y_new)
    plt.xlabel('theta(degree)', fontsize=15)
    plt.ylabel('range(meters)', fontsize=15)
    plt.savefig(r"C:\Users\PETER\PycharmProjects\levelset-ver4.3.2\experiment\frame_" + str(i) + "_accuracy.png")
    plt.show()

    ### poly_function_pre=poly_function
    ### start_bool = False

if __name__ == "__main__":
    process()