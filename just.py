#
# this is a prototype software able to recognize the rectangle laying on the table
# it can determine the angle of the rectangle and its coordinates
#
# just2, just3 is the image of 105x149mm on A4 white page
# just4 - 150x210 pink page on A3 white page with 25mm margins depicted points in the angles 
#

import cv2
import numpy as np



def r(x): return int(round(x))


def get_key_point(img,ROI):
	w = 0
	coord = np.zeros(2)
	t,b,l,r = ROI
	for j in range(t,b):
		for i in range(l,r):
			if img[j,i] == 0:
				w += 1
				coord += np.array([i,j])
	return coord / w



def cut_margins(img,top=50,bottom=60,left=50,right=20):
	img1 = cv2.imread(img,0)
	img_name = img[:-4]+'_cut.jpg'
	cv2.imwrite(img_name,img1[top:-bottom,left:-right])

	
def key_points(img):
	img1 = cv2.imread(img,0)
	blur = cv2.GaussianBlur(img1,(5,5),0)
	th1 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,6)
	margins = [(0,100,0,100),(800,850,0,100),(50,100,1100,1210),(750,850,1100,1210)]
	
	points = []
	print "image size:", np.shape(th1)
	for k in range(4):
		points.append(get_key_point(th1,margins[k]))
	
	return points


def find_homo():
	XYZs = [np.array([0., 0., 1.]), np.array([0., 297.-50., 1.]),
			np.array([420.-50., 0., 1.]), np.array([420.-50., 297.-50., 1.])]
	
	points = key_points('just4_cut.jpg')
	#print points
	r1 = np.hstack([np.zeros(3), -XYZs[0], points[0][1]*XYZs[0]])
	r2 = np.hstack([XYZs[0], np.zeros(3), -points[0][0]*XYZs[0]])
	r3 = np.hstack([-points[0][1]*XYZs[0], points[0][0]*XYZs[0], np.zeros(3)])
	A = np.vstack([r1,r2,r3])

	for i in range(1,4):
		r1 = np.hstack([np.zeros(3), -XYZs[i], points[i][1]*XYZs[i]])
		r2 = np.hstack([XYZs[i], np.zeros(3), -points[i][0]*XYZs[i]])
		r3 = np.hstack([-points[i][1]*XYZs[i], points[i][0]*XYZs[i], np.zeros(3)])
		A = np.vstack([A,r1.copy(),r2.copy(),r3.copy()])
	U,S,V = np.linalg.svd(A,full_matrices=True)
	#print V
	h = V[8]
	H = np.vstack([h[:3],h[3:6],h[6:]])
	#print "H:",H
	Hinv = np.linalg.inv(H)
	for i in range(4):
		XYZ = Hinv.dot(np.hstack([points[i],1.]))
		#print "point %i:" % i, points[i], (uv/uv[2])[:2]
		print "point %i:" % i, points[i], XYZ/XYZ[2], XYZ[2]
	
	return Hinv


def kmeans_4(lines):
	r_min = min(r for [r,th] in lines)
	r_max = max(r for [r,th] in lines)
	th_min = min(th for [r,th] in lines)
	th_max = max(th for [r,th] in lines)
	centroids = [ np.array([r,th]) for r in [r_min,r_max] for th in [th_min,th_max]]
	new_lines = []
	for [r,th] in lines:
		distances = [(r-Cr)**2 + (th-Cth)**2 for [Cr,Cth] in centroids]
		new_lines.append((np.argmin(distances),r,th))
	
	for i in range(4):
		points_i = [ (r,th) for (ind,r,th) in new_lines if ind == i]
		centroids[i] = np.array([sum(r for [r,th] in points_i), sum(th for [r,th] in points_i) ]) /len(points_i) 
	## it is better to have a few iterations

	return centroids


def get_lines(img):
	blur = cv2.GaussianBlur(img,(5,5),0)
	edges = cv2.Canny(blur,20,50)
	cv2.imshow('edges',edges)

	rho = 1.
	theta = 0.002
	threshold = 120
	lines = cv2.HoughLines(edges, rho, theta, threshold)
	lines = [ a for [a] in lines]
	
	return kmeans_4(lines)


def get_intersection(l1, l2):
	r1,th1 = l1
	r2,th2 = l2
	#print th1,th2

	c1,s1 = np.cos(th1),np.sin(th1)
	c2,s2 = np.cos(th2),np.sin(th2)

	A = np.array([[s1, -s2],
				  [-c1, c2]])
	C = np.array([ r1*c1 - r2*c2, r1*s1 - r2*s2])

	t1,t2 = np.linalg.inv(A).dot(C)
	return r1*np.array([c1,s1]) + t1*np.array([-s1,c1])


def demo():
	img = cv2.imread('just4_cut.jpg',0)
	
	Hinv = find_homo()	
	lines = get_lines(img)
	
	print "lines detected:",lines
	pxs = []
	for i in range(len(lines)):
		for j in range(len(lines)):
			if i <= j: break
			elif abs(lines[i][1]-lines[j][1]) < 0.1: continue
			pxs.append(get_intersection(lines[i],lines[j]))
	print "Pxs:", pxs
	
	for (x,y) in pxs:
		img = cv2.circle(img,(int(x),int(y)), 3, 255, -1)


	points = [ Hinv.dot(np.array([u,v,1.])) for [u,v] in pxs]
	points = [ (p/p[2])[:2] for p in points ]
	print 'Points:', points

	distances = []
	for i in range(len(points)):
		for j in range(len(points)):
			if i <= j: break
			distances.append(np.linalg.norm(points[i]-points[j]))
	distances.sort()
	print distances[:4]

	cv2.imshow('image',img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	

if __name__ == '__main__':
	Hinv = demo()


