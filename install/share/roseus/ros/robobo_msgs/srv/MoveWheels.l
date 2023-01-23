;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::MoveWheels)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'MoveWheels (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::MOVEWHEELS")
  (make-package "ROBOBO_MSGS::MOVEWHEELS"))
(unless (find-package "ROBOBO_MSGS::MOVEWHEELSREQUEST")
  (make-package "ROBOBO_MSGS::MOVEWHEELSREQUEST"))
(unless (find-package "ROBOBO_MSGS::MOVEWHEELSRESPONSE")
  (make-package "ROBOBO_MSGS::MOVEWHEELSRESPONSE"))

(in-package "ROS")

(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(if (not (find-package "STD_MSGS"))
  (ros::roseus-add-msgs "std_msgs"))


(defclass robobo_msgs::MoveWheelsRequest
  :super ros::object
  :slots (_lspeed _rspeed _time _unlockid ))

(defmethod robobo_msgs::MoveWheelsRequest
  (:init
   (&key
    ((:lspeed __lspeed) (instance std_msgs::Int8 :init))
    ((:rspeed __rspeed) (instance std_msgs::Int8 :init))
    ((:time __time) (instance std_msgs::Int32 :init))
    ((:unlockid __unlockid) (instance std_msgs::Int16 :init))
    )
   (send-super :init)
   (setq _lspeed __lspeed)
   (setq _rspeed __rspeed)
   (setq _time __time)
   (setq _unlockid __unlockid)
   self)
  (:lspeed
   (&rest __lspeed)
   (if (keywordp (car __lspeed))
       (send* _lspeed __lspeed)
     (progn
       (if __lspeed (setq _lspeed (car __lspeed)))
       _lspeed)))
  (:rspeed
   (&rest __rspeed)
   (if (keywordp (car __rspeed))
       (send* _rspeed __rspeed)
     (progn
       (if __rspeed (setq _rspeed (car __rspeed)))
       _rspeed)))
  (:time
   (&rest __time)
   (if (keywordp (car __time))
       (send* _time __time)
     (progn
       (if __time (setq _time (car __time)))
       _time)))
  (:unlockid
   (&rest __unlockid)
   (if (keywordp (car __unlockid))
       (send* _unlockid __unlockid)
     (progn
       (if __unlockid (setq _unlockid (car __unlockid)))
       _unlockid)))
  (:serialization-length
   ()
   (+
    ;; std_msgs/Int8 _lspeed
    (send _lspeed :serialization-length)
    ;; std_msgs/Int8 _rspeed
    (send _rspeed :serialization-length)
    ;; std_msgs/Int32 _time
    (send _time :serialization-length)
    ;; std_msgs/Int16 _unlockid
    (send _unlockid :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; std_msgs/Int8 _lspeed
       (send _lspeed :serialize s)
     ;; std_msgs/Int8 _rspeed
       (send _rspeed :serialize s)
     ;; std_msgs/Int32 _time
       (send _time :serialize s)
     ;; std_msgs/Int16 _unlockid
       (send _unlockid :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; std_msgs/Int8 _lspeed
     (send _lspeed :deserialize buf ptr-) (incf ptr- (send _lspeed :serialization-length))
   ;; std_msgs/Int8 _rspeed
     (send _rspeed :deserialize buf ptr-) (incf ptr- (send _rspeed :serialization-length))
   ;; std_msgs/Int32 _time
     (send _time :deserialize buf ptr-) (incf ptr- (send _time :serialization-length))
   ;; std_msgs/Int16 _unlockid
     (send _unlockid :deserialize buf ptr-) (incf ptr- (send _unlockid :serialization-length))
   ;;
   self)
  )

(defclass robobo_msgs::MoveWheelsResponse
  :super ros::object
  :slots (_error ))

(defmethod robobo_msgs::MoveWheelsResponse
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

(defclass robobo_msgs::MoveWheels
  :super ros::object
  :slots ())

(setf (get robobo_msgs::MoveWheels :md5sum-) "f7d9b6ab6c37c283e7f3f620a3f5c819")
(setf (get robobo_msgs::MoveWheels :datatype-) "robobo_msgs/MoveWheels")
(setf (get robobo_msgs::MoveWheels :request) robobo_msgs::MoveWheelsRequest)
(setf (get robobo_msgs::MoveWheels :response) robobo_msgs::MoveWheelsResponse)

(defmethod robobo_msgs::MoveWheelsRequest
  (:response () (instance robobo_msgs::MoveWheelsResponse :init)))

(setf (get robobo_msgs::MoveWheelsRequest :md5sum-) "f7d9b6ab6c37c283e7f3f620a3f5c819")
(setf (get robobo_msgs::MoveWheelsRequest :datatype-) "robobo_msgs/MoveWheelsRequest")
(setf (get robobo_msgs::MoveWheelsRequest :definition-)
      "std_msgs/Int8 lspeed
std_msgs/Int8 rspeed
std_msgs/Int32 time
std_msgs/Int16 unlockid

================================================================================
MSG: std_msgs/Int8
int8 data

================================================================================
MSG: std_msgs/Int32
int32 data
================================================================================
MSG: std_msgs/Int16
int16 data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")

(setf (get robobo_msgs::MoveWheelsResponse :md5sum-) "f7d9b6ab6c37c283e7f3f620a3f5c819")
(setf (get robobo_msgs::MoveWheelsResponse :datatype-) "robobo_msgs/MoveWheelsResponse")
(setf (get robobo_msgs::MoveWheelsResponse :definition-)
      "std_msgs/Int8 lspeed
std_msgs/Int8 rspeed
std_msgs/Int32 time
std_msgs/Int16 unlockid

================================================================================
MSG: std_msgs/Int8
int8 data

================================================================================
MSG: std_msgs/Int32
int32 data
================================================================================
MSG: std_msgs/Int16
int16 data
---
std_msgs/Int8 error


================================================================================
MSG: std_msgs/Int8
int8 data
")



(provide :robobo_msgs/MoveWheels "f7d9b6ab6c37c283e7f3f620a3f5c819")


