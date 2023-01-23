;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::SetEmotion)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'SetEmotion (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::SETEMOTION")
  (make-package "ROBOBO_MSGS::SETEMOTION"))
(unless (find-package "ROBOBO_MSGS::SETEMOTIONREQUEST")
  (make-package "ROBOBO_MSGS::SETEMOTIONREQUEST"))
(unless (find-package "ROBOBO_MSGS::SETEMOTIONRESPONSE")
  (make-package "ROBOBO_MSGS::SETEMOTIONRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::SetEmotionRequest
  :super ros::object
  :slots (_emotion ))

(defmethod robobo_msgs::SetEmotionRequest
  (:init
   (&key
    ((:emotion __emotion) (instance std_msgs::String :init))
    )
   (send-super :init)
   (setq _emotion __emotion)
   self)
  (:emotion
   (&rest __emotion)
   (if (keywordp (car __emotion))
       (send* _emotion __emotion)
     (progn
       (if __emotion (setq _emotion (car __emotion)))
       _emotion)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/String _emotion
    (send _emotion :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _emotion
       (send _emotion :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _emotion
     (send _emotion :deserialize buf ptr-) (incf ptr- (send _emotion :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::SetEmotionResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::SetEmotionResponse
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

(defclass robobo_msgs::SetEmotion
  :super ros::object
  :slots ())

(setf (get robobo_msgs::SetEmotion :md5sum-) "b56a22e94c3e6cd89d04a799670be723")
(setf (get robobo_msgs::SetEmotion :datatype-) "robobo_msgs/SetEmotion")
(setf (get robobo_msgs::SetEmotion :request) robobo_msgs::SetEmotionRequest)
(setf (get robobo_msgs::SetEmotion :response) robobo_msgs::SetEmotionResponse)

(defmethod robobo_msgs::SetEmotionRequest
  (:response () (instance robobo_msgs::SetEmotionResponse :init)))

(setf (get robobo_msgs::SetEmotionRequest :md5sum-) "b56a22e94c3e6cd89d04a799670be723")
(setf (get robobo_msgs::SetEmotionRequest :datatype-) "robobo_msgs/SetEmotionRequest")
(setf (get robobo_msgs::SetEmotionRequest :definition-)
      "std_msgs/String emotion

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::SetEmotionResponse :md5sum-) "b56a22e94c3e6cd89d04a799670be723")
(setf (get robobo_msgs::SetEmotionResponse :datatype-) "robobo_msgs/SetEmotionResponse")
(setf (get robobo_msgs::SetEmotionResponse :definition-)
      "std_msgs/String emotion

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/SetEmotion "b56a22e94c3e6cd89d04a799670be723")


