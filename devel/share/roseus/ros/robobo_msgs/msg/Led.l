;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::Led)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'Led (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::LED")
  (make-package "ROBOBO_MSGS::LED"))

(in-package "ROS")
;;//! \htmlinclude Led.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::Led
  :super ros::object
  :slots (_id _color ))

(defmethod robobo_msgs::Led
  (:init
   (&key
    ((:id __id) (instance std_msgs::String :init))
    ((:color __color) (instance std_msgs::ColorRGBA :init))
    )
   (send-super :init)
   (setq _id __id)
   (setq _color __color)
   self)
  (:id
   (&rest __id)
   (if (keywordp (car __id))
       (send* _id __id)
     (progn
       (if __id (setq _id (car __id)))
       _id)))
  (:color
   (&rest __color)
   (if (keywordp (car __color))
       (send* _color __color)
     (progn
       (if __color (setq _color (car __color)))
       _color)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _id
    (send _id :serialization-length)
    ;; std_msgs/ColorRGBA _color
    (send _color :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _id
       (send _id :serialize s)
     ;; std_msgs/ColorRGBA _color
       (send _color :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _id
     (send _id :deserialize buf ptr-) (incf ptr- (send _id :serialization-length))
   ;; std_msgs/ColorRGBA _color
     (send _color :deserialize buf ptr-) (incf ptr- (send _color :serialization-length))
   ;;
   self)
  )

(setf (get robobo_msgs::Led :md5sum-) "5da60702833e9003bbb8cd608e95a6bd")
(setf (get robobo_msgs::Led :datatype-) "robobo_msgs/Led")
(setf (get robobo_msgs::Led :definition-)
      "std_msgs/String id
std_msgs/ColorRGBA color

================================================================================
MSG: std_msgs/String
string data

================================================================================
MSG: std_msgs/ColorRGBA
float32 r
float32 g
float32 b
float32 a

")



(provide :robobo_msgs/Led "5da60702833e9003bbb8cd608e95a6bd")


