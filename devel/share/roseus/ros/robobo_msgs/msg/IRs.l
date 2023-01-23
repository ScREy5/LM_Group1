;; Auto-generated. Do not edit!


(when (boundp 'robobo_msgs::IRs)
  (if (not (find-package "ROBOBO_MSGS"))
    (make-package "ROBOBO_MSGS"))
  (shadow 'IRs (find-package "ROBOBO_MSGS")))
(unless (find-package "ROBOBO_MSGS::IRS")
  (make-package "ROBOBO_MSGS::IRS"))

(in-package "ROS")
;;//! \htmlinclude IRs.msg.html
(if (not (find-package "SENSOR_MSGS"))
  (ros::roseus-add-msgs "sensor_msgs"))


(defclass robobo_msgs::IRs
  :super ros::object
  :slots (_FrontC _FrontR _FrontRR _FrontL _FrontLL _BackC _BackR _BackL ))

(defmethod robobo_msgs::IRs
  (:init
   (&key
    ((:FrontC __FrontC) (instance sensor_msgs::Range :init))
    ((:FrontR __FrontR) (instance sensor_msgs::Range :init))
    ((:FrontRR __FrontRR) (instance sensor_msgs::Range :init))
    ((:FrontL __FrontL) (instance sensor_msgs::Range :init))
    ((:FrontLL __FrontLL) (instance sensor_msgs::Range :init))
    ((:BackC __BackC) (instance sensor_msgs::Range :init))
    ((:BackR __BackR) (instance sensor_msgs::Range :init))
    ((:BackL __BackL) (instance sensor_msgs::Range :init))
    )
   (send-super :init)
   (setq _FrontC __FrontC)
   (setq _FrontR __FrontR)
   (setq _FrontRR __FrontRR)
   (setq _FrontL __FrontL)
   (setq _FrontLL __FrontLL)
   (setq _BackC __BackC)
   (setq _BackR __BackR)
   (setq _BackL __BackL)
   self)
  (:FrontC
   (&rest __FrontC)
   (if (keywordp (car __FrontC))
       (send* _FrontC __FrontC)
     (progn
       (if __FrontC (setq _FrontC (car __FrontC)))
       _FrontC)))
  (:FrontR
   (&rest __FrontR)
   (if (keywordp (car __FrontR))
       (send* _FrontR __FrontR)
     (progn
       (if __FrontR (setq _FrontR (car __FrontR)))
       _FrontR)))
  (:FrontRR
   (&rest __FrontRR)
   (if (keywordp (car __FrontRR))
       (send* _FrontRR __FrontRR)
     (progn
       (if __FrontRR (setq _FrontRR (car __FrontRR)))
       _FrontRR)))
  (:FrontL
   (&rest __FrontL)
   (if (keywordp (car __FrontL))
       (send* _FrontL __FrontL)
     (progn
       (if __FrontL (setq _FrontL (car __FrontL)))
       _FrontL)))
  (:FrontLL
   (&rest __FrontLL)
   (if (keywordp (car __FrontLL))
       (send* _FrontLL __FrontLL)
     (progn
       (if __FrontLL (setq _FrontLL (car __FrontLL)))
       _FrontLL)))
  (:BackC
   (&rest __BackC)
   (if (keywordp (car __BackC))
       (send* _BackC __BackC)
     (progn
       (if __BackC (setq _BackC (car __BackC)))
       _BackC)))
  (:BackR
   (&rest __BackR)
   (if (keywordp (car __BackR))
       (send* _BackR __BackR)
     (progn
       (if __BackR (setq _BackR (car __BackR)))
       _BackR)))
  (:BackL
   (&rest __BackL)
   (if (keywordp (car __BackL))
       (send* _BackL __BackL)
     (progn
       (if __BackL (setq _BackL (car __BackL)))
       _BackL)))
  (:serialization-length
   ()
   (+
    ;; sensor_msgs/Range _FrontC
    (send _FrontC :serialization-length)
    ;; sensor_msgs/Range _FrontR
    (send _FrontR :serialization-length)
    ;; sensor_msgs/Range _FrontRR
    (send _FrontRR :serialization-length)
    ;; sensor_msgs/Range _FrontL
    (send _FrontL :serialization-length)
    ;; sensor_msgs/Range _FrontLL
    (send _FrontLL :serialization-length)
    ;; sensor_msgs/Range _BackC
    (send _BackC :serialization-length)
    ;; sensor_msgs/Range _BackR
    (send _BackR :serialization-length)
    ;; sensor_msgs/Range _BackL
    (send _BackL :serialization-length)
    ))
  (:serialize
   (&optional strm)
   (let ((s (if strm strm
              (make-string-output-stream (send self :serialization-length)))))
     ;; sensor_msgs/Range _FrontC
       (send _FrontC :serialize s)
     ;; sensor_msgs/Range _FrontR
       (send _FrontR :serialize s)
     ;; sensor_msgs/Range _FrontRR
       (send _FrontRR :serialize s)
     ;; sensor_msgs/Range _FrontL
       (send _FrontL :serialize s)
     ;; sensor_msgs/Range _FrontLL
       (send _FrontLL :serialize s)
     ;; sensor_msgs/Range _BackC
       (send _BackC :serialize s)
     ;; sensor_msgs/Range _BackR
       (send _BackR :serialize s)
     ;; sensor_msgs/Range _BackL
       (send _BackL :serialize s)
     ;;
     (if (null strm) (get-output-stream-string s))))
  (:deserialize
   (buf &optional (ptr- 0))
   ;; sensor_msgs/Range _FrontC
     (send _FrontC :deserialize buf ptr-) (incf ptr- (send _FrontC :serialization-length))
   ;; sensor_msgs/Range _FrontR
     (send _FrontR :deserialize buf ptr-) (incf ptr- (send _FrontR :serialization-length))
   ;; sensor_msgs/Range _FrontRR
     (send _FrontRR :deserialize buf ptr-) (incf ptr- (send _FrontRR :serialization-length))
   ;; sensor_msgs/Range _FrontL
     (send _FrontL :deserialize buf ptr-) (incf ptr- (send _FrontL :serialization-length))
   ;; sensor_msgs/Range _FrontLL
     (send _FrontLL :deserialize buf ptr-) (incf ptr- (send _FrontLL :serialization-length))
   ;; sensor_msgs/Range _BackC
     (send _BackC :deserialize buf ptr-) (incf ptr- (send _BackC :serialization-length))
   ;; sensor_msgs/Range _BackR
     (send _BackR :deserialize buf ptr-) (incf ptr- (send _BackR :serialization-length))
   ;; sensor_msgs/Range _BackL
     (send _BackL :deserialize buf ptr-) (incf ptr- (send _BackL :serialization-length))
   ;;
   self)
  )

(setf (get robobo_msgs::IRs :md5sum-) "3cda0da1598fe1278d12814909cfb3d7")
(setf (get robobo_msgs::IRs :datatype-) "robobo_msgs/IRs")
(setf (get robobo_msgs::IRs :definition-)
      "sensor_msgs/Range FrontC
sensor_msgs/Range FrontR
sensor_msgs/Range FrontRR
sensor_msgs/Range FrontL
sensor_msgs/Range FrontLL
sensor_msgs/Range BackC
sensor_msgs/Range BackR
sensor_msgs/Range BackL

================================================================================
MSG: sensor_msgs/Range
# Single range reading from an active ranger that emits energy and reports
# one range reading that is valid along an arc at the distance measured. 
# This message is  not appropriate for laser scanners. See the LaserScan
# message if you are working with a laser scanner.

# This message also can represent a fixed-distance (binary) ranger.  This
# sensor will have min_range===max_range===distance of detection.
# These sensors follow REP 117 and will output -Inf if the object is detected
# and +Inf if the object is outside of the detection range.

Header header           # timestamp in the header is the time the ranger
                        # returned the distance reading

# Radiation type enums
# If you want a value added to this list, send an email to the ros-users list
uint8 ULTRASOUND=0
uint8 INFRARED=1

uint8 radiation_type    # the type of radiation used by the sensor
                        # (sound, IR, etc) [enum]

float32 field_of_view   # the size of the arc that the distance reading is
                        # valid for [rad]
                        # the object causing the range reading may have
                        # been anywhere within -field_of_view/2 and
                        # field_of_view/2 at the measured range. 
                        # 0 angle corresponds to the x-axis of the sensor.

float32 min_range       # minimum range value [m]
float32 max_range       # maximum range value [m]
                        # Fixed distance rangers require min_range==max_range

float32 range           # range data [m]
                        # (Note: values < range_min or > range_max
                        # should be discarded)
                        # Fixed distance rangers only output -Inf or +Inf.
                        # -Inf represents a detection within fixed distance.
                        # (Detection too close to the sensor to quantify)
                        # +Inf represents no detection within the fixed distance.
                        # (Object out of range)
================================================================================
MSG: std_msgs/Header
# Standard metadata for higher-level stamped data types.
# This is generally used to communicate timestamped data 
# in a particular coordinate frame.
# 
# sequence ID: consecutively increasing ID 
uint32 seq
#Two-integer timestamp that is expressed as:
# * stamp.sec: seconds (stamp_secs) since epoch (in Python the variable is called 'secs')
# * stamp.nsec: nanoseconds since stamp_secs (in Python the variable is called 'nsecs')
# time-handling sugar is provided by the client library
time stamp
#Frame this data is associated with
string frame_id

")



(provide :robobo_msgs/IRs "3cda0da1598fe1278d12814909cfb3d7")


