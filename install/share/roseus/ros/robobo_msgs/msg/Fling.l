;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::Fling)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'Fling (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::FLING")
  (make-package "ROBOBO_MSGS::FLING"))

(in-package "ROS")
;;//! \htmlinclude Fling.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::Fling
  :super ros::object
  :slots (_angle _time _distance ))

(defmethod robobo_msgs::Fling
  (:init
   (&key
    ((:angle __angle) (instance std_msgs::Int16 :init))
    ((:time __time) (instance std_msgs::Int32 :init))
    ((:distance __distance) (instance std_msgs::Int16 :init))
    )
   (send-super :init)
   (setq _angle __angle)
   (setq _time __time)
   (setq _distance __distance)
   self)
  (:angle
   (&rest __angle)
   (if (keywordp (car __angle))
       (send* _angle __angle)
     (progn
       (if __angle (setq _angle (car __angle)))
       _angle)))
  (:time
   (&rest __time)
   (if (keywordp (car __time))
       (send* _time __time)
     (progn
       (if __time (setq _time (car __time)))
       _time)))
  (:distance
   (&rest __distance)
   (if (keywordp (car __distance))
       (send* _distance __distance)
     (progn
       (if __distance (setq _distance (car __distance)))
       _distance)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int16 _angle
    (send _angle :serialization-length)
    ;; std_msgs/Int32 _time
    (send _time :serialization-length)
    ;; std_msgs/Int16 _distance
    (send _distance :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int16 _angle
       (send _angle :serialize s)
     ;; std_msgs/Int32 _time
       (send _time :serialize s)
     ;; std_msgs/Int16 _distance
       (send _distance :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int16 _angle
     (send _angle :deserialize buf ptr-) (incf ptr- (send _angle :serialization-length))
   ;; std_msgs/Int32 _time
     (send _time :deserialize buf ptr-) (incf ptr- (send _time :serialization-length))
   ;; std_msgs/Int16 _distance
     (send _distance :deserialize buf ptr-) (incf ptr- (send _distance :serialization-length))
   ;;
   self)
  )

(setf (get robobo_msgs::Fling :md5sum-) "223606e2db34b4f54252a5c13a50f185")
(setf (get robobo_msgs::Fling :datatype-) "robobo_msgs/Fling")
(setf (get robobo_msgs::Fling :definition-)
      "std_msgs/Int16 angle
std_msgs/Int32 time
std_msgs/Int16 distance

================================================================================
MSG: std_msgs/Int16
int16 data

================================================================================
MSG: std_msgs/Int32
int32 data
")



(provide :robobo_msgs/Fling "223606e2db34b4f54252a5c13a50f185")


