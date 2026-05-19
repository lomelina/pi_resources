from picamera2 import Picamera2
import cv2
import numpy as np
from hailo_platform import VDevice, FormatType
from face_detector import FaceDetector

HEF_PATH = "scrfd_2.5g.hef"
TIMEOUT_MS = 1000
FRAME_SIZE = (640, 640)

def process_frame(frame, configured_model, bindings, output_buffers):
    model_input = np.ascontiguousarray(frame, dtype=np.uint8)
    bindings.input().set_buffer(model_input)
    configured_model.run([bindings], TIMEOUT_MS)
    return output_buffers


def main():
    face_detector = FaceDetector()

    with VDevice() as vdevice:
        model = vdevice.create_infer_model(HEF_PATH)
        model.set_batch_size(1)
        model.input().set_format_type(FormatType.UINT8)
        for output_info in model.outputs:
            model.output(output_info.name).set_format_type(FormatType.FLOAT32)

        with model.configure() as configured_model:
            output_buffers = {
                info.name: np.empty(info.shape, dtype=np.float32)
                for info in model.outputs
            }
            bindings = configured_model.create_bindings(output_buffers=output_buffers)
            configured_model.activate()

            picam2 = None
            try:
                picam2 = Picamera2()
                preview_config = picam2.create_preview_configuration(
                    main={"size": FRAME_SIZE, "format": "RGB888"}
                )
                picam2.configure(preview_config)
                picam2.start()

                while True:
                    frame = picam2.capture_array()
                    outputs = process_frame(
                        frame,
                        configured_model,
                        bindings,
                        output_buffers,
                    )
                    faces = face_detector.find_faces(outputs)

                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR, frame)
                    
                    for x1, y1, x2, y2, score, keypoints in faces:
                        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                        for x, y in keypoints:
                            cv2.circle(frame, (int(x), int(y)), 3, (0, 0, 255), -1)

                    display_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    cv2.imshow("PiCamera2 Preview", display_frame)
                    if cv2.waitKey(1) == 27:  # press ESC to quit
                        break
            except KeyboardInterrupt:
                pass
            finally:
                if picam2 is not None:
                    picam2.stop()
                configured_model.deactivate()
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
