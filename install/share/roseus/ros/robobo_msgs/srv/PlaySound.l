;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::PlaySound)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'PlaySound (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::PLAYSOUND")
  (make-package "ROBOBO_MSGS::PLAYSOUND"))
(unless (find-package "ROBOBO_MSGS::PLAYSOUNDREQUEST")
  (make-package "ROBOBO_MSGS::PLAYSOUNDREQUEST"))
(unless (find-package "ROBOBO_MSGS::PLAYSOUNDRESPONSE")
  (make-package "ROBOBO_MSGS::PLAYSOUNDRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::PlaySoundRequest
  :super ros::object
  :slots (_sound ))

(defmethod robobo_msgs::PlaySoundRequest
  (:init
   (&key
    ((:sound __sound) (instance std_msgs::String :init))
    )
   (send-super :init)
   (setq _sound __sound)
   self)
  (:sound
   (&rest __sound)
   (if (keywordp (car __sound))
       (send* _sound __sound)
     (progn
       (if __sound (setq _sound (car __sound)))
       _sound)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _sound
    (send _sound :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _sound
       (send _sound :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _sound
     (send _sound :deserialize buf ptr-) (incf ptr- (send _sound :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::PlaySoundResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::PlaySoundResponse
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

(defclass robobo_msgs::PlaySound
  :super ros::object
  :slots ())

(setf (get robobo_msgs::PlaySound :md5sum-) "c95eacdcbf61c615623911c64d3b591f")
(setf (get robobo_msgs::PlaySound :datatype-) "robobo_msgs/PlaySound")
(setf (get robobo_msgs::PlaySound :request) robobo_msgs::PlaySoundRequest)
(setf (get robobo_msgs::PlaySound :response) robobo_msgs::PlaySoundResponse)

(defmethod robobo_msgs::PlaySoundRequest
  (:response () (instance robobo_msgs::PlaySoundResponse :init)))

(setf (get robobo_msgs::PlaySoundRequest :md5sum-) "c95eacdcbf61c615623911c64d3b591f")
(setf (get robobo_msgs::PlaySoundRequest :datatype-) "robobo_msgs/PlaySoundRequest")
(setf (get robobo_msgs::PlaySoundRequest :definition-)
      "std_msgs/String sound

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::PlaySoundResponse :md5sum-) "c95eacdcbf61c615623911c64d3b591f")
(setf (get robobo_msgs::PlaySoundResponse :datatype-) "robobo_msgs/PlaySoundResponse")
(setf (get robobo_msgs::PlaySoundResponse :definition-)
      "std_msgs/String sound

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/PlaySound "c95eacdcbf61c615623911c64d3b591f")


