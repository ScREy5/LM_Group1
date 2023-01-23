;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::Talk)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'Talk (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::TALK")
  (make-package "ROBOBO_MSGS::TALK"))
(unless (find-package "ROBOBO_MSGS::TALKREQUEST")
  (make-package "ROBOBO_MSGS::TALKREQUEST"))
(unless (find-package "ROBOBO_MSGS::TALKRESPONSE")
  (make-package "ROBOBO_MSGS::TALKRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::TalkRequest
  :super ros::object
  :slots (_text ))

(defmethod robobo_msgs::TalkRequest
  (:init
   (&key
    ((:text __text) (instance std_msgs::String :init))
    )
   (send-super :init)
   (setq _text __text)
   self)
  (:text
   (&rest __text)
   (if (keywordp (car __text))
       (send* _text __text)
     (progn
       (if __text (setq _text (car __text)))
       _text)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _text
    (send _text :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _text
       (send _text :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _text
     (send _text :deserialize buf ptr-) (incf ptr- (send _text :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::TalkResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::TalkResponse
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

(defclass robobo_msgs::Talk
  :super ros::object
  :slots ())

(setf (get robobo_msgs::Talk :md5sum-) "d8fdbf7e3675ec38cfc5f64bfa512f70")
(setf (get robobo_msgs::Talk :datatype-) "robobo_msgs/Talk")
(setf (get robobo_msgs::Talk :request) robobo_msgs::TalkRequest)
(setf (get robobo_msgs::Talk :response) robobo_msgs::TalkResponse)

(defmethod robobo_msgs::TalkRequest
  (:response () (instance robobo_msgs::TalkResponse :init)))

(setf (get robobo_msgs::TalkRequest :md5sum-) "d8fdbf7e3675ec38cfc5f64bfa512f70")
(setf (get robobo_msgs::TalkRequest :datatype-) "robobo_msgs/TalkRequest")
(setf (get robobo_msgs::TalkRequest :definition-)
      "std_msgs/String text

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::TalkResponse :md5sum-) "d8fdbf7e3675ec38cfc5f64bfa512f70")
(setf (get robobo_msgs::TalkResponse :datatype-) "robobo_msgs/TalkResponse")
(setf (get robobo_msgs::TalkResponse :definition-)
      "std_msgs/String text

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/Talk "d8fdbf7e3675ec38cfc5f64bfa512f70")


