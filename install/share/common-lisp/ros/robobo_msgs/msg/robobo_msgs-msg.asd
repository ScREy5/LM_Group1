
(cl:in-package :asdf)

(defsystem "robobo_msgs-msg"
  :depends-on (:roslisp-msg-protocol :roslisp-utils :sensor_msgs-msg
               :std_msgs-msg
)
  :components ((:file "_package")
    (:file "Fling" :depends-on ("_package_Fling"))
    (:file "_package_Fling" :depends-on ("_package"))
    (:file "IRs" :depends-on ("_package_IRs"))
    (:file "_package_IRs" :depends-on ("_package"))
    (:file "Led" :depends-on ("_package_Led"))
    (:file "_package_Led" :depends-on ("_package"))
    (:file "Tap" :depends-on ("_package_Tap"))
    (:file "_package_Tap" :depends-on ("_package"))
    (:file "Wheels" :depends-on ("_package_Wheels"))
    (:file "_package_Wheels" :depends-on ("_package"))
  ))