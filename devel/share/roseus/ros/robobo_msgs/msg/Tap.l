;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::Tap)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'Tap (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::TAP")
  (make-package "ROBOBO_MSGS::TAP"))

(in-package "ROS")
;;//! \htmlinclude Tap.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::Tap
  :super ros::object
  :slots (_x _y ))

(defmethod robobo_msgs::Tap
  (:init
   (&key
    ((:x __x) (instance std_msgs::Int8 :init))
    ((:y __y) (instance std_msgs::Int8 :init))
    )
   (send-super :init)
   (setq _x __x)
   (setq _y __y)
   self)
  (:x
   (&rest __x)
   (if (keywordp (car __x))
       (send* _x __x)
     (progn
       (if __x (setq _x (car __x)))
       _x)))
  (:y
   (&rest __y)
   (if (keywordp (car __y))
       (send* _y __y)
     (progn
       (if __y (setq _y (car __y)))
       _y)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8 _x
    (send _x :serialization-length)
    ;; std_msgs/Int8 _y
    (send _y :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8 _x
       (send _x :serialize s)
     ;; std_msgs/Int8 _y
       (send _y :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8 _x
     (send _x :deserialize buf ptr-) (incf ptr- (send _x :serialization-length))
   ;; std_msgs/Int8 _y
     (send _y :deserialize buf ptr-) (incf ptr- (send _y :serialization-length))
   ;;
   self)
  )

(setf (get robobo_msgs::Tap :md5sum-) "c455a05a0b8c9dc9827ed72d23a4de17")
(setf (get robobo_msgs::Tap :datatype-) "robobo_msgs/Tap")
(setf (get robobo_msgs::Tap :definition-)
      "std_msgs/Int8 x
std_msgs/Int8 y

================================================================================
MSG: std_msgs/Int8
int8 data

")



(provide :robobo_msgs/Tap "c455a05a0b8c9dc9827ed72d23a4de17")


