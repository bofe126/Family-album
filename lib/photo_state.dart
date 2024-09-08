import 'package:flutter/foundation.dart';
import 'package:image_picker/image_picker.dart';
import 'face_recognition_service.dart';

class PhotoState extends ChangeNotifier {
  List<XFile> _imageFileList = [];
  Map<String, List<FaceData>> _detectedFaces = {};
  List<FaceData> _uniqueFaces = [];
  FaceData? _selectedFace;

  List<XFile> get imageFileList => _imageFileList;
  List<FaceData> get uniqueFaces => _uniqueFaces;
  FaceData? get selectedFace => _selectedFace;

  List<XFile> get filteredImages {
    if (_selectedFace == null) return _imageFileList;
    return _imageFileList.where((image) => 
      _detectedFaces[image.path]?.any((face) => _compareFaces(face, _selectedFace!)) ?? false
    ).toList();
  }

  bool faceDetected(String imagePath) {
    return _detectedFaces.containsKey(imagePath) && _detectedFaces[imagePath]!.isNotEmpty;
  }

  void addImages(List<XFile> newImages) async {
    _imageFileList.addAll(newImages);
    await _detectFaces(newImages);
    notifyListeners();
  }

  Future<void> _detectFaces(List<XFile> images) async {
    for (var image in images) {
      List<FaceData> faces = await FaceRecognitionService.recognizeFaces([image.path]);
      _detectedFaces[image.path] = faces;
      for (var face in faces) {
        if (!_uniqueFaces.any((uniqueFace) => _compareFaces(face, uniqueFace))) {
          _uniqueFaces.add(face);
        }
      }
    }
  }

  void selectFace(FaceData face) {
    _selectedFace = face;
    notifyListeners();
  }

  bool _compareFaces(FaceData face1, FaceData face2) {
    return FaceRecognitionService.compareFaces(face1.features, face2.features) > 0.7;
  }
}