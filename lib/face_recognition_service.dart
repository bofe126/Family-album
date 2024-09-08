import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:isolate';
import 'dart:typed_data';

typedef DetectFacesC = Int32 Function(Pointer<Utf8> imagePath, Pointer<Int32> faces, Int32 maxFaces, Pointer<Pointer<Uint8>> faceData, Pointer<Int32> faceDataSizes);
typedef DetectFacesDart = int Function(Pointer<Utf8> imagePath, Pointer<Int32> faces, int maxFaces, Pointer<Pointer<Uint8>> faceData, Pointer<Int32> faceDataSizes);

class FaceRecognitionService {
  static late DynamicLibrary _lib;
  static late DetectFacesDart _detectFaces;
  static bool _isInitialized = false;

  static Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      String dllPath = 'build\\windows\\x64\\runner\\Debug\\face_recognition.dll';
      print('Attempting to load DLL from: $dllPath');
      if (!File(dllPath).existsSync()) {
        throw FileSystemException('DLL file not found', dllPath);
      }
      _lib = Platform.isWindows
          ? DynamicLibrary.open(dllPath)
          : DynamicLibrary.process();
      print('Successfully loaded face_recognition.dll');
      
      _detectFaces = _lib.lookup<NativeFunction<DetectFacesC>>('detect_faces').asFunction();
      print('Successfully initialized _detectFaces');
      
      _isInitialized = true;
      print('FaceRecognitionService initialization complete');
    } catch (e) {
      print('Failed to initialize FaceRecognitionService: $e');
      rethrow;
    }
  }

  static bool get isInitialized => _isInitialized;

  static Future<List<Uint8List>> recognizeFaces(List<String> imagePaths) async {
    final p = ReceivePort();
    await Isolate.spawn(_isolateRecognizeFaces, [p.sendPort, imagePaths]);
    final result = await p.first;
    if (result is List<Uint8List>) {
      return result;
    } else {
      throw StateError('Unexpected result from isolate');
    }
  }

  static void _isolateRecognizeFaces(List<dynamic> args) async {
    SendPort sendPort = args[0];
    List<String> imagePaths = args[1];
    List<Uint8List> faceImages = [];

    try {
      await initialize();  // Re-initialize in the isolate
      
      if (!_isInitialized) {
        throw StateError('FaceRecognitionService has not been initialized');
      }

      for (String imagePath in imagePaths) {
        try {
          print('Processing image: $imagePath');
          final normalizedPath = imagePath.replaceAll('\\', '/');
          final imagePathPtr = normalizedPath.toNativeUtf8();
          final maxFaces = 100;
          final facesPtr = calloc<Int32>(maxFaces * 4); // 每张脸4个int
          final faceDataPtr = calloc<Pointer<Uint8>>(maxFaces); // 最多maxFaces张脸的图像数据
          final faceDataSizesPtr = calloc<Int32>(maxFaces); // 每张脸图像数据的大小

          print('Calling _detectFaces');
          final numFaces = _detectFaces(imagePathPtr, facesPtr, maxFaces, faceDataPtr, faceDataSizesPtr);
          print('_detectFaces returned: $numFaces faces');

          for (int i = 0; i < numFaces; i++) {
            final x = facesPtr[i*4];
            final y = facesPtr[i*4+1];
            final width = facesPtr[i*4+2];
            final height = facesPtr[i*4+3];
            
            print('Face $i: ($x, $y), size: ${width}x$height');
            
            final dataSize = faceDataSizesPtr[i];
            print('Face $i data size: $dataSize');
            final faceImageData = faceDataPtr[i].asTypedList(dataSize);
            faceImages.add(Uint8List.fromList(faceImageData));

            calloc.free(faceDataPtr[i]);
          }

          calloc.free(imagePathPtr);
          calloc.free(facesPtr);
          calloc.free(faceDataPtr);
          calloc.free(faceDataSizesPtr);
        } catch (e, stackTrace) {
          print('Error processing image $imagePath: $e');
          print('Stack trace: $stackTrace');
        }
      }
    } catch (e, stackTrace) {
      print('Error in isolate: $e');
      print('Stack trace: $stackTrace');
    } finally {
      sendPort.send(faceImages);
    }
  }
}