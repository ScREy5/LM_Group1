;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::Wheels)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'Wheels (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::WHEELS")
  (make-package "ROBOBO_MSGS::WHEELS"))

(in-package "ROS")
;;//! \htmlinclude Wheels.msg.html
(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::Wheels
  :super ros::object
  :slots (_wheelPosR _wheelPosL _wheelSpeedR _wheelSpeedL ))

(defmethod robobo_msgs::Wheels
  (:init
   (&key
    ((:wheelPosR __wheelPosR) (instance std_msgs::Int64 :init))
    ((:wheelPosL __wheelPosL) (instance std_msgs::Int64 :init))
    ((:wheelSpeedR __wheelSpeedR) (instance std_msgs::Int64 :init))
    ((:wheelSpeedL __wheelSpeedL) (instance std_msgs::Int64 :init))
    )
   (send-super :init)
   (setq _wheelPosR __wheelPosR)
   (setq _wheelPosL __wheelPosL)
   (setq _wheelSpeedR __wheelSpeedR)
   (setq _wheelSpeedL __wheelSpeedL)
   self)
  (:wheelPosR
   (&rest __wheelPosR)
   (if (keywordp (car __wheelPosR))
       (send* _wheelPosR __wheelPosR)
     (progn
       (if __wheelPosR (setq _wheelPosR (car __wheelPosR)))
       _wheelPosR)))
  (:wheelPosL
   (&rest __wheelPosL)
   (if (keywordp (car __wheelPosL))
       (send* _wheelPosL __wheelPosL)
     (progn
       (if __wheelPosL (setq _wheelPosL (car __wheelPosL)))
       _wheelPosL)))
  (:wheelSpeedR
   (&rest __wheelSpeedR)
   (if (keywordp (car __wheelSpeedR))
       (send* _wheelSpeedR __wheelSpeedR)
     (progn
       (if __wheelSpeedR (setq _wheelSpeedR (car __wheelSpeedR)))
       _wheelSpeedR)))
  (:wheelSpeedL
   (&rest __wheelSpeedL)
   (if (keywordp (car __wheelSpeedL))
       (send* _wheelSpeedL __wheelSpeedL)
     (progn
       (if __wheelSpeedL (setq _wheelSpeedL (car __wheelSpeedL)))
       _wheelSpeedL)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int64 _wheelPosR
    (send _wheelPosR :serialization-length)
    ;; std_msgs/Int64 _wheelPosL
    (send _wheelPosL :serialization-length)
    ;; std_msgs/Int64 _wheelSpeedR
    (send _wheelSpeedR :serialization-length)
    ;; std_msgs/Int64 _wheelSpeedL
    (send _wheelSpeedL :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int64 _wheelPosR
       (send _wheelPosR :serialize s)
     ;; std_msgs/Int64 _wheelPosL
       (send _wheelPosL :serialize s)
     ;; std_msgs/Int64 _wheelSpeedR
       (send _wheelSpeedR :serialize s)
     ;; std_msgs/Int64 _wheelSpeedL
       (send _wheelSpeedL :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int64 _wheelPosR
     (send _wheelPosR :deserialize buf ptr-) (incf ptr- (send _wheelPosR :serialization-length))
   ;; std_msgs/Int64 _wheelPosL
     (send _wheelPosL :deserialize buf ptr-) (incf ptr- (send _wheelPosL :serialization-length))
   ;; std_msgs/Int64 _wheelSpeedR
     (send _wheelSpeedR :deserialize buf ptr-) (incf ptr- (send _wheelSpeedR :serialization-length))
   ;; std_msgs/Int64 _wheelSpeedL
     (send _wheelSpeedL :deserialize buf ptr-) (incf ptr- (send _wheelSpeedL :serialization-length))
   ;;
   self)
  )

(setf (get robobo_msgs::Wheels :md5sum-) "e3007c5c2ec7faa3f8f71bd3a0d670c5")
(setf (get robobo_msgs::Wheels :datatype-) "robobo_msgs/Wheels")
(setf (get robobo_msgs::Wheels :definition-)
      "std_msgs/Int64 wheelPosR
std_msgs/Int64 wheelPosL
std_msgs/Int64 wheelSpeedR
std_msgs/Int64 wheelSpeedL

================================================================================
MSG: std_msgs/Int64
int64 data
")



(provide :robobo_msgs/Wheels "e3007c5c2ec7faa3f8f71bd3a0d670c5")


