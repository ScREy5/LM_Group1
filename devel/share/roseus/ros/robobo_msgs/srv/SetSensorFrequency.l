;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::SetSensorFrequency)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'SetSensorFrequency (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::SETSENSORFREQUENCY")
  (make-package "ROBOBO_MSGS::SETSENSORFREQUENCY"))
(unless (find-package "ROBOBO_MSGS::SETSENSORFREQUENCYREQUEST")
  (make-package "ROBOBO_MSGS::SETSENSORFREQUENCYREQUEST"))
(unless (find-package "ROBOBO_MSGS::SETSENSORFREQUENCYRESPONSE")
  (make-package "ROBOBO_MSGS::SETSENSORFREQUENCYRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::SetSensorFrequencyRequest
  :super ros::object
  :slots (_frequency ))

(defmethod robobo_msgs::SetSensorFrequencyRequest
  (:init
   (&key
    ((:frequency __frequency) (instance std_msgs::Int8 :init))
    )
   (send-super :init)
   (setq _frequency __frequency)
   self)
  (:frequency
   (&rest __frequency)
   (if (keywordp (car __frequency))
       (send* _frequency __frequency)
     (progn
       (if __frequency (setq _frequency (car __frequency)))
       _frequency)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8 _frequency
    (send _frequency :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8 _frequency
       (send _frequency :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8 _frequency
     (send _frequency :deserialize buf ptr-) (incf ptr- (send _frequency :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::SetSensorFrequencyResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::SetSensorFrequencyResponse
  (:init
   (&key
    ((:error __error) (instance std_msgs::Int8 :init))
    )
   (send-super :init)
   (setq _error __error)
   self)
  (:error
   (&rest __error)
   (if (keywordp (car __error))
       (send* _error __error)
     (progn
       (if __error (setq _error (car __error)))
       _error)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8 _error
    (send _error :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8 _error
       (send _error :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8 _error
     (send _error :deserialize buf ptr-) (incf ptr- (send _error :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::SetSensorFrequency
  :super ros::object
  :slots ())

(setf (get robobo_msgs::SetSensorFrequency :md5sum-) "ea3a75b016df1ca9629d8a0e272b0401")
(setf (get robobo_msgs::SetSensorFrequency :datatype-) "robobo_msgs/SetSensorFrequency")
(setf (get robobo_msgs::SetSensorFrequency :request) robobo_msgs::SetSensorFrequencyRequest)
(setf (get robobo_msgs::SetSensorFrequency :response) robobo_msgs::SetSensorFrequencyResponse)

(defmethod robobo_msgs::SetSensorFrequencyRequest
  (:response () (instance robobo_msgs::SetSensorFrequencyResponse :init)))

(setf (get robobo_msgs::SetSensorFrequencyRequest :md5sum-) "ea3a75b016df1ca9629d8a0e272b0401")
(setf (get robobo_msgs::SetSensorFrequencyRequest :datatype-) "robobo_msgs/SetSensorFrequencyRequest")
(setf (get robobo_msgs::SetSensorFrequencyRequest :definition-)
      "std_msgs/Int8 frequency   #0 for LOW
                          #1 for NORMAL
                          #2 for HIGH
                          #3 for MAX

================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::SetSensorFrequencyResponse :md5sum-) "ea3a75b016df1ca9629d8a0e272b0401")
(setf (get robobo_msgs::SetSensorFrequencyResponse :datatype-) "robobo_msgs/SetSensorFrequencyResponse")
(setf (get robobo_msgs::SetSensorFrequencyResponse :definition-)
      "std_msgs/Int8 frequency   #0 for LOW
                          #1 for NORMAL
                          #2 for HIGH
                          #3 for MAX

================================================================================
MSG: std_msgs/Int8
int8 data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/SetSensorFrequency "ea3a75b016df1ca9629d8a0e272b0401")


