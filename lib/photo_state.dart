import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'face_recognition_service.dart';
import 'dart:math';

class PhotoState extends ChangeNotifier {
  List<XFile> _imageFileList = [];
  Map<String, List<FaceData>> _detectedFaces = {};
  List<UniqueFace> _uniqueFaces = [];
  UniqueFace? _selectedFace;

  List<XFile> get imageFileList => _imageFileList;
  List<UniqueFace> get uniqueFaces => _uniqueFaces;
  UniqueFace? get selectedFace => _selectedFace;

  List<XFile> get filteredImages {
    if (_selectedFace == null) return _imageFileList;
    return _selectedFace!.images;
  }

  bool faceDetected(String imagePath) {
    return _detectedFaces.containsKey(imagePath) && _detectedFaces[imagePath]!.isNotEmpty;
  }

  void addImages(List<XFile> newImages) async {
    _imageFileList.addAll(newImages);
    await _detectFaces(newImages);
    _clusterFaces();
    notifyListeners();
  }

  Future<void> _detectFaces(List<XFile> images) async {
    for (var image in images) {
      List<FaceData> faces = await FaceRecognitionService.recognizeFaces([image.path]);
      _detectedFaces[image.path] = faces;
      for (var face in faces) {
        _uniqueFaces.add(UniqueFace(face, [image]));
      }
    }
  }

  void _clusterFaces() {
    List<UniqueFace> newUniqueFaces = [];
    for (var face in _uniqueFaces) {
      bool added = false;
      for (var uniqueFace in newUniqueFaces) {
        if (_compareFaces(face.faceData, uniqueFace.faceData)) {
          uniqueFace.addImage(face.images.first);
          added = true;
          print('Face added to existing cluster. Features: ${visualizeFeatures(face.faceData.features)}');
          break;
        }
      }
      if (!added) {
        newUniqueFaces.add(face);
        print('New unique face added. Features: ${visualizeFeatures(face.faceData.features)}');
      }
    }
    _uniqueFaces = newUniqueFaces;
    print("Number of unique faces: ${_uniqueFaces.length}");
  }

  void selectFace(UniqueFace face) {
    _selectedFace = face;
    notifyListeners();
  }

  bool _compareFaces(FaceData face1, FaceData face2) {
    double similarity = FaceRecognitionService.compareFaces(face1.features, face2.features);
    print('Face comparison similarity: $similarity');
    return similarity > 0.85; // 调整这个阈值，余弦相似度通常使用更高的阈值
  }

  // 添加一个方法来可视化特征向量
  String visualizeFeatures(Float32List features) {
    return features.sublist(0, 10).map((f) => f.toStringAsFixed(2)).join(', ') + '...';
  }
}

class UniqueFace {
  final FaceData faceData;
  final List<XFile> images;

  UniqueFace(this.faceData, this.images);

  void addImage(XFile image) {
    if (!images.contains(image)) {
      images.add(image);
    }
  }
}