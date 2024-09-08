import 'dart:ffi';
import 'dart:io';
import 'package:ffi/ffi.dart';
import 'dart:isolate';
import 'dart:typed_data';
import 'package:flutter/services.dart' show rootBundle;
import 'package:path_provider/path_provider.dart';

typedef DetectFacesC = Int32 Function(
    Pointer<Utf8> imagePath,
    Pointer<Utf8> prototxtPath,
    Pointer<Utf8> caffeModelPath,
    Pointer<Utf8> arcfaceModelPath,
    Pointer<Int32> faces,
    Int32 maxFaces,
    Pointer<Pointer<Uint8>> faceData,
    Pointer<Int32> faceDataSizes,
    Pointer<Pointer<Float>> faceFeatures);

typedef DetectFacesDart = int Function(
    Pointer<Utf8> imagePath,
    Pointer<Utf8> prototxtPath,
    Pointer<Utf8> caffeModelPath,
    Pointer<Utf8> arcfaceModelPath,
    Pointer<Int32> faces,
    int maxFaces,
    Pointer<Pointer<Uint8>> faceData,
    Pointer<Int32> faceDataSizes,
    Pointer<Pointer<Float>> faceFeatures);

typedef CompareFacesC = Float Function(
    Pointer<Float> features1,
    Pointer<Float> features2,
    Int32 featureSize);

typedef CompareFacesDart = double Function(
    Pointer<Float> features1,
    Pointer<Float> features2,
    int featureSize);

class FaceRecognitionService {
  static late DynamicLibrary _lib;
  static late DetectFacesDart _detectFaces;
  static late CompareFacesDart _compareFaces;
  static bool _isInitialized = false;
  static late String _dllPath;
  static late String _prototxtPath;
  static late String _caffeModelPath;
  static late String _arcfaceModelPath;

  static Future<void> initialize() async {
    if (_isInitialized) return;
    try {
      await _copyAssetsToLocal();
      _dllPath = 'assets/face_recognition.dll';
      _prototxtPath = await _getLocalPath('deploy.prototxt');
      _caffeModelPath = await _getLocalPath('res10_300x300_ssd_iter_140000.caffemodel');
      _arcfaceModelPath = await _getLocalPath('arcface_model.onnx');

      _lib = DynamicLibrary.open(_dllPath);
      _detectFaces = _lib.lookup<NativeFunction<DetectFacesC>>('detect_faces').asFunction();
      _compareFaces = _lib.lookup<NativeFunction<CompareFacesC>>('compare_faces').asFunction();
      
      _isInitialized = true;
    } catch (e) {
      print('Failed to initialize FaceRecognitionService: $e');
      rethrow;
    }
  }

  static Future<void> _copyAssetsToLocal() async {
    final assets = [
      'deploy.prototxt',
      'res10_300x300_ssd_iter_140000.caffemodel',
      'arcface_model.onnx'
    ];

    for (final asset in assets) {
      final localPath = await _getLocalPath(asset);
      if (!File(localPath).existsSync()) {
        try {
          final data = await rootBundle.load('assets/$asset');
          await File(localPath).writeAsBytes(data.buffer.asUint8List());
          print('Successfully copied $asset to $localPath');
        } catch (e) {
          print('Failed to copy $asset: $e');
          rethrow;
        }
      } else {
        print('$asset already exists at $localPath');
      }
    }
  }

  static Future<String> _getLocalPath(String fileName) async {
    final directory = await getApplicationDocumentsDirectory();
    return '${directory.path}/$fileName';
  }

  static bool get isInitialized => _isInitialized;

  static Future<List<FaceData>> recognizeFaces(List<String> imagePaths) async {
    if (!_isInitialized) {
      throw StateError('FaceRecognitionService has not been initialized');
    }

    final p = ReceivePort();
    await Isolate.spawn(_isolateRecognizeFaces, [p.sendPort, imagePaths, _dllPath, _prototxtPath, _caffeModelPath, _arcfaceModelPath]);
    final result = await p.first;
    if (result is List<FaceData>) {
      // 添加日志输出
      for (var face in result) {
        print('Face detected: ${face.features.length} features, first 5 values: ${face.features.sublist(0, 5)}');
      }
      return result;
    } else {
      throw StateError('Unexpected result from isolate');
    }
  }

  static void _isolateRecognizeFaces(List<dynamic> args) {
    SendPort sendPort = args[0];
    List<String> imagePaths = args[1];
    String dllPath = args[2];
    String prototxtPath = args[3];
    String caffeModelPath = args[4];
    String arcfaceModelPath = args[5];
    
    List<FaceData> faceDataList = [];

    final lib = DynamicLibrary.open(dllPath);
    final detectFaces = lib.lookup<NativeFunction<DetectFacesC>>('detect_faces').asFunction<DetectFacesDart>();

    for (String imagePath in imagePaths) {
      try {
        print('Processing image: $imagePath');
        final normalizedPath = imagePath.replaceAll('\\', '/');
        final imagePathPtr = normalizedPath.toNativeUtf8();
        final prototxtPathPtr = prototxtPath.toNativeUtf8();
        final caffeModelPathPtr = caffeModelPath.toNativeUtf8();
        final arcfaceModelPathPtr = arcfaceModelPath.toNativeUtf8();
        final maxFaces = 100;
        final facesPtr = calloc<Int32>(maxFaces * 4);
        final faceDataPtr = calloc<Pointer<Uint8>>(maxFaces);
        final faceDataSizesPtr = calloc<Int32>(maxFaces);
        final featurePtrs = calloc<Pointer<Float>>(maxFaces);

        final numFaces = detectFaces(imagePathPtr, prototxtPathPtr, caffeModelPathPtr, arcfaceModelPathPtr, facesPtr, maxFaces, faceDataPtr, faceDataSizesPtr, featurePtrs);
        print('detectFaces returned: $numFaces faces');

        for (int i = 0; i < numFaces; i++) {
          final dataSize = faceDataSizesPtr[i];
          final faceImageData = faceDataPtr[i].asTypedList(dataSize);
          final featureData = featurePtrs[i].asTypedList(512);
          faceDataList.add(FaceData(
            faceImage: Uint8List.fromList(faceImageData),
            features: Float32List.fromList(featureData),
          ));
          calloc.free(faceDataPtr[i]);
          calloc.free(featurePtrs[i]);
        }

        calloc.free(imagePathPtr);
        calloc.free(prototxtPathPtr);
        calloc.free(caffeModelPathPtr);
        calloc.free(arcfaceModelPathPtr);
        calloc.free(facesPtr);
        calloc.free(faceDataPtr);
        calloc.free(faceDataSizesPtr);
        calloc.free(featurePtrs);
      } catch (e) {
        print('Error processing image $imagePath: $e');
      }
    }

    sendPort.send(faceDataList);
  }

  static double compareFaces(Float32List features1, Float32List features2) {
    final ptr1 = calloc<Float>(features1.length);
    final ptr2 = calloc<Float>(features2.length);

    for (int i = 0; i < features1.length; i++) {
      ptr1[i] = features1[i];
      ptr2[i] = features2[i];
    }

    final similarity = _compareFaces(ptr1, ptr2, features1.length);

    calloc.free(ptr1);
    calloc.free(ptr2);

    return similarity;
  }
}

class FaceData {
  final Uint8List faceImage;
  final Float32List features;

  FaceData({required this.faceImage, required this.features});
}