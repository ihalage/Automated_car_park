# Automated_car_park

*****License Plate Recognition*****
*****Vehicle tracking*****
*****Parking violation identification and empty parking slot detection*****
*****Deep Learning*****


System explained:

Deep neural network based license plate recognition. One deep neural network is used for image segmentation. It extracts only the number plate area of the vehicle. This neural network is implemented using caffe framework. The other deep neural network which is implemented using tensorflow library is used for character recognition. 

Once the license plate is regognized we check for that vehicle number in our database. If it exists, we let the vehicle to enter the car park. Or else the security personnel can give a manual overide to the database using our GUI. Once the vehicle enters the car park the vehicle tracking system gets initialized. The vehicle will be tracked from the entrance to its destination, while it is parked until it exits the car park. For vehicle identification we use tiny yolo. And for tracking we use correlation tracker. Our system is capable of tracking multiple vehicles over multiple overlapping cameras.

Once the vehicle is parked, we check for parking violations. This is achieved using a background substraction method. If there is any violation, we notify the driver using our andriod app. And also at this stage we identify empty parking slots as well. The driver can request a parking slot at the gate using our android app. If there is a request, we provide the route to an empty parking slot using our android app. 


