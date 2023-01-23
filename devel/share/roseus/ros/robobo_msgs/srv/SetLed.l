;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::SetLed)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'SetLed (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::SETLED")
  (make-package "ROBOBO_MSGS::SETLED"))
(unless (find-package "ROBOBO_MSGS::SETLEDREQUEST")
  (make-package "ROBOBO_MSGS::SETLEDREQUEST"))
(unless (find-package "ROBOBO_MSGS::SETLEDRESPONSE")
  (make-package "ROBOBO_MSGS::SETLEDRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::SetLedRequest
  :super ros::object
  :slots (_id _color ))

(defmethod robobo_msgs::SetLedRequest
  (:init
   (&key
    ((:id __id) (instance std_msgs::String :init))
    ((:color __color) (instance std_msgs::String :init))
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
    ;; std_msgs/String _color
    (send _color :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/String _id
       (send _id :serialize s)
     ;; std_msgs/String _color
       (send _color :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/String _id
     (send _id :deserialize buf ptr-) (incf ptr- (send _id :serialization-length))
   ;; std_msgs/String _color
     (send _color :deserialize buf ptr-) (incf ptr- (send _color :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::SetLedResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::SetLedResponse
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

(defclass robobo_msgs::SetLed
  :super ros::object
  :slots ())

(setf (get robobo_msgs::SetLed :md5sum-) "cae286d48c8a0ad77943140106a4ad56")
(setf (get robobo_msgs::SetLed :datatype-) "robobo_msgs/SetLed")
(setf (get robobo_msgs::SetLed :request) robobo_msgs::SetLedRequest)
(setf (get robobo_msgs::SetLed :response) robobo_msgs::SetLedResponse)

(defmethod robobo_msgs::SetLedRequest
  (:response () (instance robobo_msgs::SetLedResponse :init)))

(setf (get robobo_msgs::SetLedRequest :md5sum-) "cae286d48c8a0ad77943140106a4ad56")
(setf (get robobo_msgs::SetLedRequest :datatype-) "robobo_msgs/SetLedRequest")
(setf (get robobo_msgs::SetLedRequest :definition-)
      "std_msgs/String id
std_msgs/String color

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::SetLedResponse :md5sum-) "cae286d48c8a0ad77943140106a4ad56")
(setf (get robobo_msgs::SetLedResponse :datatype-) "robobo_msgs/SetLedResponse")
(setf (get robobo_msgs::SetLedResponse :definition-)
      "std_msgs/String id
std_msgs/String color

================================================================================
MSG: std_msgs/String
string data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/SetLed "cae286d48c8a0ad77943140106a4ad56")


