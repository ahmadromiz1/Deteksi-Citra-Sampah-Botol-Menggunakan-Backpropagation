-- phpMyAdmin SQL Dump
-- version 5.1.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 29 Jan 2022 pada 13.00
-- Versi server: 10.4.22-MariaDB
-- Versi PHP: 8.1.1

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `botol_ta`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `botol`
--

CREATE TABLE `botol` (
  `id` int(11) NOT NULL,
  `b` double NOT NULL,
  `g` double NOT NULL,
  `r` double NOT NULL,
  `label` int(12) NOT NULL,
  `file` varchar(30) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

--
-- Dumping data untuk tabel `botol`
--

INSERT INTO `botol` (`id`, `b`, `g`, `r`, `label`, `file`) VALUES
(1, 0.38961613, 0.38983297, 0.3898562, 0, '0001_B.png'),
(2, 0.39117306, 0.39168432, 0.3920484, 0, '0002_B.png'),
(3, 0.39159912, 0.3918315, 0.39206392, 0, '0003_B.png'),
(4, 0.39135122, 0.3916379, 0.39203292, 0, '0004_B.png'),
(5, 0.3914984, 0.3917153, 0.39193994, 0, '0005_B.png'),
(6, 0.39135897, 0.39172307, 0.39203292, 0, '0006_B.png'),
(7, 0.39135125, 0.39158362, 0.39202517, 0, '0007_B.png'),
(8, 0.3911576, 0.39166883, 0.39196318, 0, '0008_B.png'),
(9, 0.39116535, 0.39141318, 0.39191672, 0, '0009_B.png'),
(10, 0.3911034, 0.3913977, 0.39193222, 0, '0010_B.png'),
(11, 0.39121956, 0.3916069, 0.3919012, 0, '0011_B.png'),
(12, 0.39078578, 0.39138225, 0.39197868, 0, '0012_B.png'),
(13, 0.39161462, 0.3917773, 0.39210266, 0, '0013_B.png'),
(14, 0.39121956, 0.39161462, 0.3918935, 0, '0014_B.png'),
(15, 0.39149845, 0.39175403, 0.39190122, 0, '0015_B.png'),
(16, 0.3909097, 0.39125055, 0.3918083, 0, '0016_B.png'),
(17, 0.39092517, 0.39145967, 0.391878, 0, '0017_B.png'),
(18, 0.3916069, 0.39180055, 0.39197865, 0, '0018_B.png'),
(19, 0.39106464, 0.39140546, 0.3919709, 0, '0019_B.png'),
(20, 0.39087874, 0.39149845, 0.39191672, 0, '0020_B.png'),
(21, 0.39116538, 0.39156818, 0.39198646, 0, '0021_B.png'),
(22, 0.39084, 0.391297, 0.39187795, 0, '0022_B.png'),
(23, 0.39109564, 0.39135125, 0.3916766, 0, '0023_B.png'),
(24, 0.39201745, 0.39198646, 0.39200193, 0, '0024_B.png'),
(25, 0.39200193, 0.39200196, 0.3920019, 0, '0025_B.png'),
(26, 0.39196318, 0.39207163, 0.3920794, 0, '0026_B.png'),
(27, 0.39194, 0.3920329, 0.39204842, 0, '0027_B.png'),
(28, 0.39200193, 0.39204067, 0.3920639, 0, '0028_B.png'),
(29, 0.39209488, 0.39211038, 0.39198643, 0, '0029_B.png'),
(30, 0.3919554, 0.3920252, 0.39202517, 0, '0030_B.png'),
(31, 0.39188573, 0.39193222, 0.39182374, 0, '0031_B.png'),
(32, 0.39212587, 0.39212587, 0.3921259, 0, '0032_B.png'),
(33, 0.39207166, 0.3921104, 0.39213365, 0, '0033_B.png'),
(34, 0.3921259, 0.39214137, 0.39212587, 0, '0034_B.png'),
(35, 0.3918935, 0.39193222, 0.39180052, 0, '0035_B.png'),
(36, 0.39208716, 0.39208716, 0.39208716, 0, '0036_B.png'),
(37, 0.39211813, 0.39211816, 0.39211813, 0, '0037_B.png'),
(38, 0.39210263, 0.39210263, 0.39210263, 0, '0038_B.png'),
(39, 0.3920407, 0.39207166, 0.39204067, 0, '0039_B.png'),
(40, 0.39192444, 0.39192447, 0.39190125, 0, '0040_B.png'),
(41, 0.3920949, 0.39211038, 0.39207166, 0, '0041_B.png'),
(42, 0.39180824, 0.39197093, 0.39190897, 0, '0042_B.png'),
(43, 0.3920407, 0.3920949, 0.39202517, 0, '0043_B.png'),
(44, 0.39203295, 0.39211813, 0.3921104, 0, '0044_B.png'),
(45, 0.39211816, 0.39212587, 0.39211813, 0, '0045_B.png'),
(46, 0.39207166, 0.39208716, 0.39210266, 0, '0046_B.png'),
(47, 0.39210266, 0.39211813, 0.39212584, 0, '0047_B.png'),
(48, 0.39206392, 0.3920794, 0.39208713, 0, '0048_B.png'),
(49, 0.39210263, 0.39211038, 0.39213365, 0, '0049_B.png'),
(50, 0.3920794, 0.39211813, 0.39211813, 0, '0050_B.png'),
(51, 0.39204067, 0.39210266, 0.3921414, 0, '0051_B.png'),
(52, 0.39209488, 0.39209488, 0.39211813, 0, '0052_B.png'),
(53, 0.39208713, 0.39208713, 0.39209494, 0, '0053_B.png'),
(54, 0.39200968, 0.39201748, 0.39198643, 0, '0054_B.png'),
(55, 0.3921027, 0.39211038, 0.3921104, 0, '0055_B.png'),
(56, 0.39197093, 0.39202517, 0.3919942, 0, '0056_B.png'),
(57, 0.39193994, 0.39201745, 0.39199418, 0, '0057_B.png'),
(58, 0.3920484, 0.3920794, 0.39207166, 0, '0058_B.png'),
(59, 0.39197096, 0.39206398, 0.3920407, 0, '0059_B.png'),
(60, 0.39211038, 0.3921259, 0.39213362, 0, '0060_B.png'),
(61, 0.39197868, 0.3920407, 0.39209494, 0, '0061_B.png'),
(62, 0.39204067, 0.39206392, 0.3920949, 0, '0062_B.png'),
(63, 0.39211813, 0.39213362, 0.39213362, 0, '0063_B.png'),
(64, 0.39208716, 0.39211813, 0.39214912, 0, '0064_B.png'),
(65, 0.39190122, 0.39195544, 0.39200196, 0, '0065_B.png'),
(66, 0.3921259, 0.39213362, 0.3921414, 0, '0066_B.png'),
(67, 0.39203292, 0.39206392, 0.39210266, 0, '0067_B.png'),
(68, 0.39213365, 0.3921336, 0.39213362, 0, '0068_B.png'),
(69, 0.39197093, 0.39200193, 0.39197865, 0, '0069_B.png'),
(70, 0.39204845, 0.39207938, 0.39207938, 0, '0070_B.png'),
(71, 0.39193222, 0.39207938, 0.39208713, 0, '0071_B.png'),
(72, 0.3920562, 0.39208716, 0.39214137, 0, '0072_B.png'),
(73, 0.39199415, 0.39209494, 0.39202517, 0, '0073_B.png'),
(74, 0.39208716, 0.39210266, 0.3921104, 0, '0074_B.png'),
(75, 0.39213365, 0.3921414, 0.39214137, 0, '0075_B.png'),
(76, 0.3917076, 0.39198643, 0.3920252, 0, '0076_B.png'),
(77, 0.39073157, 0.39135125, 0.39125055, 0, '0077_B.png'),
(78, 0.39213365, 0.3920872, 0.39207166, 0, '0078_B.png'),
(79, 0.39203295, 0.39203295, 0.39200965, 0, '0079_B.png'),
(80, 0.39205617, 0.39208713, 0.39204842, 0, '0080_B.png'),
(81, 0.39206392, 0.39203292, 0.39198643, 0, '0081_B.png'),
(82, 0.39205617, 0.39211816, 0.39201745, 0, '0082_B.png'),
(83, 0.39205617, 0.39206395, 0.39205617, 0, '0083_B.png'),
(84, 0.39206395, 0.39211038, 0.39204842, 0, '0084_B.png'),
(85, 0.39209488, 0.39212587, 0.39208716, 0, '0085_B.png'),
(86, 0.39204845, 0.39204067, 0.39201745, 0, '0086_B.png'),
(87, 0.39209488, 0.39201745, 0.39204067, 0, '0087_B.png'),
(88, 0.3920407, 0.3920794, 0.39204845, 0, '0088_B.png'),
(89, 0.3921336, 0.39198646, 0.39209488, 0, '0089_B.png'),
(90, 0.39214137, 0.39204842, 0.39207938, 0, '0090_B.png'),
(91, 0.39212587, 0.3920949, 0.39207166, 0, '0091_B.png'),
(92, 0.39205617, 0.39187798, 0.39204067, 0, '0092_B.png'),
(93, 0.3920717, 0.39207163, 0.39202517, 0, '0093_B.png'),
(94, 0.3919632, 0.39198643, 0.39193997, 0, '0094_B.png'),
(95, 0.39212587, 0.39208716, 0.39204067, 0, '0095_B.png'),
(96, 0.39212587, 0.39210263, 0.39208716, 0, '0096_B.png'),
(97, 0.39182377, 0.3918625, 0.3917773, 0, '0097_B.png'),
(98, 0.39209494, 0.39208716, 0.39207938, 0, '0098_B.png'),
(99, 0.39198646, 0.3920717, 0.39197093, 0, '0099_B.png'),
(100, 0.39211816, 0.39210266, 0.3921104, 0, '0100_B.png'),
(101, 0.39210266, 0.39209488, 0.39203295, 0, '0101_B.png'),
(102, 0.39207166, 0.39203292, 0.39200193, 0, '0102_B.png'),
(103, 0.39212587, 0.39211813, 0.39211816, 0, '0103_B.png'),
(104, 0.3920794, 0.39207944, 0.39204067, 0, '0104_B.png'),
(105, 0.3921104, 0.39204848, 0.39204842, 0, '0105_B.png'),
(106, 0.39190122, 0.391909, 0.39174634, 0, '0106_B.png'),
(107, 0.3921259, 0.39206392, 0.3920794, 0, '0107_B.png'),
(108, 0.39204067, 0.39201748, 0.39197096, 0, '0108_B.png'),
(109, 0.39206392, 0.39205617, 0.39203295, 0, '0109_B.png'),
(110, 0.3920484, 0.39200193, 0.39198643, 0, '0110_B.png'),
(111, 0.39204845, 0.39203295, 0.39195547, 0, '0111_B.png'),
(112, 0.39211038, 0.39209494, 0.39206392, 0, '0112_B.png'),
(113, 0.39205617, 0.39202517, 0.3919864, 0, '0113_B.png'),
(114, 0.39215684, 0.39213362, 0.39210263, 0, '0114_B.png'),
(115, 0.39211816, 0.39206395, 0.3920794, 0, '0115_B.png'),
(116, 0.3920407, 0.39210266, 0.39201745, 0, '0116_B.png'),
(117, 0.39205617, 0.39204842, 0.39203295, 0, '0117_B.png'),
(118, 0.3921414, 0.39206395, 0.3920794, 0, '0118_B.png'),
(119, 0.39203295, 0.3920097, 0.39194772, 0, '0119_B.png'),
(120, 0.39205617, 0.39206395, 0.39204842, 0, '0120_B.png'),
(121, 0.3921414, 0.3921104, 0.3921104, 0, '0121_B.png'),
(122, 0.39211044, 0.39210266, 0.39207938, 0, '0122_B.png'),
(123, 0.39212587, 0.39204842, 0.3920174, 0, '0123_B.png'),
(124, 0.3921259, 0.39206392, 0.3920097, 0, '0124_B.png'),
(125, 0.39208716, 0.39202523, 0.39200193, 0, '0125_B.png'),
(126, 0.39212584, 0.39204842, 0.39194772, 0, '0126_B.png'),
(127, 0.39209488, 0.39201745, 0.39200193, 0, '0127_B.png'),
(128, 0.3921491, 0.39208716, 0.3920639, 0, '0128_B.png'),
(129, 0.39212587, 0.39207166, 0.39201742, 0, '0129_B.png'),
(130, 0.39214912, 0.39214137, 0.39213362, 0, '0130_B.png'),
(131, 0.39210263, 0.3919864, 0.39200193, 0, '0131_B.png'),
(132, 0.39205617, 0.39203295, 0.39203295, 0, '0132_B.png'),
(133, 0.39208716, 0.39203295, 0.39205617, 0, '0133_B.png'),
(134, 0.39206392, 0.39202523, 0.39193222, 0, '0134_B.png'),
(135, 0.39211813, 0.39207166, 0.39207944, 0, '0135_B.png'),
(136, 0.39200965, 0.39196318, 0.39192447, 0, '0136_B.png'),
(137, 0.39213365, 0.39214137, 0.39212587, 0, '0137_B.png'),
(138, 0.39214137, 0.39211038, 0.39210266, 0, '0138_B.png'),
(139, 0.39210263, 0.3920949, 0.3920794, 0, '0139_B.png'),
(140, 0.39213362, 0.39207166, 0.3921104, 0, '0140_B.png'),
(141, 0.39213362, 0.39208713, 0.39205617, 0, '0141_B.png'),
(142, 0.39208713, 0.3920329, 0.39200965, 0, '0142_B.png'),
(143, 0.39211813, 0.39202517, 0.39202517, 0, '0143_B.png'),
(144, 0.39214137, 0.39211038, 0.3921104, 0, '0144_B.png'),
(145, 0.39211813, 0.39211813, 0.39210266, 0, '0145_B.png'),
(146, 0.3920949, 0.39210263, 0.39205617, 0, '0146_B.png'),
(147, 0.39214137, 0.39208713, 0.39211038, 0, '0147_B.png'),
(148, 0.39214912, 0.3920949, 0.39207166, 0, '0148_B.png'),
(149, 0.39212587, 0.3921259, 0.3921182, 0, '0149_B.png'),
(150, 0.39204842, 0.39208716, 0.39202517, 0, '0150_B.png'),
(151, 0.39210266, 0.3920794, 0.39206392, 0, '0151_B.png'),
(152, 0.39208716, 0.39204845, 0.39205617, 0, '0152_B.png'),
(153, 0.3920717, 0.3920872, 0.39205617, 0, '0153_B.png'),
(154, 0.39211816, 0.39207166, 0.39204842, 0, '0154_B.png'),
(155, 0.39204842, 0.39206395, 0.39209488, 0, '0155_B.png'),
(156, 0.39214915, 0.39214137, 0.39213362, 0, '0156_B.png'),
(157, 0.39212584, 0.39211038, 0.39210263, 0, '0157_B.png'),
(158, 0.39207944, 0.3921259, 0.39208716, 0, '0158_B.png'),
(159, 0.3921259, 0.39206392, 0.39209488, 0, '0159_B.png'),
(160, 0.39212584, 0.39209488, 0.39208716, 0, '0160_B.png'),
(161, 0.39215687, 0.39213362, 0.39213365, 0, '0161_B.png'),
(162, 0.3919322, 0.39197096, 0.39191672, 0, '0162_B.png'),
(163, 0.39213362, 0.39211038, 0.39207944, 0, '0163_B.png'),
(164, 0.39213365, 0.39208716, 0.39207163, 0, '0164_B.png'),
(165, 0.3921259, 0.3920794, 0.39207166, 0, '0165_B.png'),
(166, 0.39212593, 0.39209494, 0.3921027, 0, '0166_B.png'),
(167, 0.39208716, 0.39203295, 0.39201745, 0, '0167_B.png'),
(168, 0.39213365, 0.39213362, 0.39210263, 0, '0168_B.png'),
(169, 0.39215687, 0.39214912, 0.3921569, 0, '0169_B.png'),
(170, 0.3921104, 0.39203292, 0.39198643, 0, '0170_B.png'),
(171, 0.39206392, 0.39206395, 0.3920949, 0, '0171_B.png'),
(172, 0.3921414, 0.39211038, 0.39211816, 0, '0172_B.png'),
(173, 0.39214912, 0.39211038, 0.39211038, 0, '0173_B.png'),
(174, 0.39211038, 0.39207166, 0.39206395, 0, '0174_B.png'),
(175, 0.39210266, 0.39210266, 0.3921104, 0, '0175_B.png'),
(176, 0.39214912, 0.3921414, 0.39212587, 0, '0176_B.png'),
(177, 0.39212587, 0.3921027, 0.39208713, 0, '0177_B.png'),
(178, 0.39214137, 0.3920794, 0.39210266, 0, '0178_B.png'),
(179, 0.38989493, 0.39064634, 0.39093292, 0, '0179_B.png'),
(180, 0.39031324, 0.39104915, 0.39113435, 0, '0180_B.png'),
(181, 0.39022803, 0.39104137, 0.3912738, 0, '0181_B.png'),
(182, 0.3881288, 0.39013505, 0.39056888, 0, '0182_B.png'),
(183, 0.38949987, 0.3908555, 0.39101815, 0, '0183_B.png'),
(184, 0.39034423, 0.3915139, 0.3915217, 0, '0184_B.png'),
(185, 0.38895762, 0.3908942, 0.39051467, 0, '0185_B.png'),
(186, 0.38698235, 0.39137447, 0.39149842, 0, '0186_B.png'),
(187, 0.38061485, 0.3843873, 0.3846352, 0, '0187_B.png'),
(188, 0.3877957, 0.39102593, 0.39161465, 0, '0188_B.png'),
(189, 0.3868816, 0.3911111, 0.3918625, 0, '0189_B.png'),
(190, 0.39215687, 0.39215687, 0.39215684, 1, '0178_XB.png'),
(191, 0.3921569, 0.39215687, 0.39215687, 1, '0179_XB.png'),
(192, 0.39215687, 0.39215687, 0.39215687, 1, '0180_XB.png'),
(193, 0.39215684, 0.39215687, 0.39215684, 1, '0181_XB.png'),
(194, 0.39215687, 0.39215684, 0.39215684, 1, '0182_XB.png'),
(195, 0.39200968, 0.39169207, 0.39186248, 1, '0183_XB.png'),
(196, 0.391816, 0.3916766, 0.39194772, 1, '0184_XB.png'),
(197, 0.39197093, 0.39153716, 0.39190122, 1, '0185_XB.png'),
(198, 0.39198643, 0.39168438, 0.39183927, 1, '0186_XB.png'),
(199, 0.39215687, 0.39210266, 0.39215687, 1, '0187_XB.png'),
(200, 0.39212587, 0.39192447, 0.39211035, 1, '0188_XB.png'),
(201, 0.39210263, 0.3919787, 0.3921104, 1, '0189_XB.png'),
(202, 0.39215684, 0.39215687, 0.3921569, 1, '0190_XB.png'),
(203, 0.39215687, 0.39207166, 0.39214137, 1, '0191_XB.png'),
(204, 0.39214137, 0.39194772, 0.3920794, 1, '0192_XB.png'),
(205, 0.39200965, 0.39171535, 0.391878, 1, '0193_XB.png'),
(206, 0.39215687, 0.3920949, 0.39215687, 1, '0194_XB.png'),
(207, 0.39214137, 0.39203292, 0.39214134, 1, '0195_XB.png'),
(208, 0.39208716, 0.39204845, 0.39214137, 1, '0196_XB.png'),
(209, 0.39214912, 0.39207944, 0.39214137, 1, '0197_XB.png'),
(210, 0.3921569, 0.39214912, 0.39215687, 1, '0198_XB.png'),
(211, 0.39214912, 0.39193997, 0.3921414, 1, '0199_XB.png'),
(212, 0.39200968, 0.39171532, 0.3920562, 1, '0200_XB.png'),
(213, 0.39210266, 0.39195544, 0.39215687, 1, '0201_XB.png'),
(214, 0.39211038, 0.39203295, 0.39213362, 1, '0202_XB.png'),
(215, 0.3921259, 0.39191672, 0.39212587, 1, '0203_XB.png'),
(216, 0.39193222, 0.3916379, 0.39214137, 1, '0204_XB.png'),
(217, 0.39213365, 0.39194772, 0.3921491, 1, '0205_XB.png'),
(218, 0.3921569, 0.39184746, 0.39215687, 1, '0206_XB.png'),
(219, 0.39213362, 0.39186248, 0.39214915, 1, '0207_XB.png'),
(220, 0.39209494, 0.39174628, 0.3921491, 1, '0208_XB.png'),
(221, 0.39214137, 0.39183924, 0.39209488, 1, '0209_XB.png'),
(222, 0.39214915, 0.39197096, 0.39213362, 1, '0210_XB.png'),
(223, 0.39211813, 0.39171532, 0.39214137, 1, '0211_XB.png'),
(224, 0.3921259, 0.3920252, 0.39214912, 1, '0212_XB.png'),
(225, 0.39213362, 0.39197096, 0.39213362, 1, '0213_XB.png'),
(226, 0.3921181, 0.39211038, 0.39214915, 1, '0214_XB.png'),
(227, 0.39213365, 0.3919632, 0.39214912, 1, '0215_XB.png'),
(228, 0.39214137, 0.39193222, 0.39211813, 1, '0216_XB.png'),
(229, 0.39210266, 0.39203295, 0.39211816, 1, '0217_XB.png'),
(230, 0.39211813, 0.39197093, 0.39212584, 1, '0218_XB.png'),
(231, 0.39212584, 0.39193222, 0.39213365, 1, '0219_XB.png'),
(232, 0.3921181, 0.39196318, 0.39214912, 1, '0220_XB.png'),
(233, 0.3920794, 0.3919167, 0.3920949, 1, '0221_XB.png'),
(234, 0.39212587, 0.39212587, 0.39214912, 1, '0222_XB.png'),
(235, 0.3920949, 0.3917773, 0.39212587, 1, '0223_XB.png'),
(236, 0.39210266, 0.39169982, 0.39211038, 1, '0224_XB.png'),
(237, 0.3921491, 0.39209488, 0.3921569, 1, '0225_XB.png'),
(238, 0.39215687, 0.39207163, 0.39215687, 1, '0226_XB.png'),
(239, 0.39215684, 0.39213362, 0.39215684, 1, '0227_XB.png'),
(240, 0.39213365, 0.3919322, 0.39213365, 1, '0228_XB.png'),
(241, 0.39214915, 0.39210263, 0.3921491, 1, '0229_XB.png'),
(242, 0.3921259, 0.3920562, 0.39215687, 1, '0230_XB.png'),
(243, 0.39213365, 0.39202517, 0.39214915, 1, '0231_XB.png'),
(244, 0.39214912, 0.39211038, 0.39214915, 1, '0232_XB.png'),
(245, 0.39214137, 0.39193222, 0.3921491, 1, '0233_XB.png'),
(246, 0.39215687, 0.39215684, 0.39215687, 1, '0234_XB.png'),
(247, 0.3921569, 0.39215687, 0.39215684, 1, '0235_XB.png'),
(248, 0.3921569, 0.39215687, 0.39215687, 1, '0236_XB.png'),
(249, 0.39215687, 0.3921568, 0.39215687, 1, '0237_XB.png'),
(250, 0.39215687, 0.3921569, 0.3921569, 1, '0238_XB.png'),
(251, 0.39215687, 0.39215684, 0.3921569, 1, '0239_XB.png'),
(252, 0.39215687, 0.39215687, 0.3921569, 1, '0240_XB.png'),
(253, 0.39215687, 0.39215684, 0.3921569, 1, '0241_XB.png'),
(254, 0.3921569, 0.39215687, 0.39215687, 1, '0242_XB.png'),
(255, 0.39215687, 0.39215687, 0.3921569, 1, '0243_XB.png'),
(256, 0.39215687, 0.39215687, 0.39215687, 1, '0244_XB.png'),
(257, 0.3921569, 0.39215687, 0.39215687, 1, '0245_XB.png'),
(258, 0.39215687, 0.3921569, 0.39215687, 1, '0246_XB.png'),
(259, 0.39215684, 0.3921569, 0.39215687, 1, '0247_XB.png'),
(260, 0.3921569, 0.39215687, 0.3921569, 1, '0248_XB.png'),
(261, 0.39215693, 0.39215687, 0.39215687, 1, '0249_XB.png'),
(262, 0.39215687, 0.39215684, 0.39215687, 1, '0250_XB.png'),
(263, 0.3921569, 0.3921569, 0.39215687, 1, '0251_XB.png'),
(264, 0.39215684, 0.39215687, 0.3921569, 1, '0252_XB.png'),
(265, 0.39156815, 0.39107236, 0.39170757, 1, '0253_XB.png'),
(266, 0.39207166, 0.39181602, 0.39165336, 1, '0254_XB.png'),
(267, 0.39181602, 0.39101818, 0.39171532, 1, '0255_XB.png'),
(268, 0.3916224, 0.39127377, 0.39150617, 1, '0256_XB.png'),
(269, 0.39200968, 0.39084, 0.3904759, 1, '0257_XB.png'),
(270, 0.3911576, 0.39104915, 0.3919477, 1, '0258_XB.png'),
(271, 0.39144418, 0.39084, 0.39143646, 1, '0259_XB.png'),
(272, 0.39190122, 0.39131254, 0.3916766, 1, '0260_XB.png'),
(273, 0.39149684, 0.39106366, 0.39068204, 1, '0261_XB.png'),
(274, 0.39186248, 0.39142868, 0.3906386, 1, '0262_XB.png'),
(275, 0.39180824, 0.39149842, 0.39125052, 1, '0263_XB.png'),
(276, 0.39204842, 0.39167663, 0.39123508, 1, '0264_XB.png'),
(277, 0.39208716, 0.39173082, 0.3909639, 1, '0265_XB.png'),
(278, 0.39207166, 0.3917618, 0.3903907, 1, '0266_XB.png'),
(279, 0.39212587, 0.39113435, 0.39027452, 1, '0267_XB.png'),
(280, 0.39211038, 0.39161465, 0.39125827, 1, '0268_XB.png'),
(281, 0.3917076, 0.3914132, 0.39071608, 1, '0269_XB.png'),
(282, 0.39168432, 0.3909872, 0.39080128, 1, '0270_XB.png'),
(283, 0.3920252, 0.39180827, 0.39094067, 1, '0271_XB.png'),
(284, 0.39211813, 0.39188573, 0.39089423, 1, '0272_XB.png'),
(285, 0.3919787, 0.39150617, 0.39112663, 1, '0273_XB.png'),
(286, 0.39127377, 0.3912738, 0.3909562, 1, '0274_XB.png'),
(287, 0.39193222, 0.39183927, 0.39114988, 1, '0275_XB.png'),
(288, 0.39187798, 0.3915449, 0.39072382, 1, '0276_XB.png'),
(289, 0.39200193, 0.391878, 0.3906386, 1, '0277_XB.png'),
(290, 0.3916766, 0.3915914, 0.39143643, 1, '0278_XB.png'),
(291, 0.3917928, 0.39149845, 0.39109564, 1, '0279_XB.png'),
(292, 0.39189348, 0.3915604, 0.3904759, 1, '0280_XB.png'),
(293, 0.39194, 0.3921027, 0.3919012, 1, '0281_XB.png'),
(294, 0.39207938, 0.39210266, 0.39155266, 1, '0282_XB.png'),
(295, 0.39214137, 0.39212587, 0.3915139, 1, '0283_XB.png'),
(296, 0.39181602, 0.39192447, 0.3915527, 1, '0284_XB.png'),
(297, 0.391359, 0.39169207, 0.39125827, 1, '0285_XB.png'),
(298, 0.39175406, 0.39184698, 0.39058438, 1, '0286_XB.png'),
(299, 0.39195544, 0.39198643, 0.39131254, 1, '0287_XB.png'),
(300, 0.39107236, 0.39152944, 0.39045265, 1, '0288_XB.png'),
(301, 0.39212587, 0.39213362, 0.39153716, 1, '0289_XB.png'),
(302, 0.3916766, 0.39185473, 0.39176953, 1, '0290_XB.png'),
(303, 0.39187798, 0.39183924, 0.39046815, 1, '0291_XB.png'),
(304, 0.39173084, 0.39059216, 0.38851613, 1, '0292_XB.png'),
(305, 0.39196318, 0.39191672, 0.3909794, 1, '0293_XB.png'),
(306, 0.39166886, 0.39165333, 0.39090198, 1, '0294_XB.png'),
(307, 0.39163786, 0.3915217, 0.39035198, 1, '0295_XB.png'),
(308, 0.39215687, 0.3921491, 0.3918935, 1, '0296_XB.png'),
(309, 0.39183927, 0.39190122, 0.39198643, 1, '0297_XB.png'),
(310, 0.39197123, 0.39079553, 0.39058927, 1, '0298_XB.png'),
(311, 0.39213622, 0.39199185, 0.3910946, 1, '0299_XB.png'),
(312, 0.39214656, 0.39209503, 0.3916515, 1, '0300_XB.png'),
(313, 0.39189902, 0.39086774, 0.3901664, 1, '0301_XB.png'),
(314, 0.39208463, 0.39176497, 0.3913112, 1, '0302_XB.png'),
(315, 0.39210528, 0.39177528, 0.39001176, 1, '0303_XB.png'),
(316, 0.39212593, 0.39194027, 0.39109465, 1, '0304_XB.png'),
(317, 0.3920125, 0.3909193, 0.3904036, 1, '0305_XB.png'),
(318, 0.39215687, 0.39215684, 0.39215687, 1, '0306_XB.png'),
(319, 0.39215687, 0.39215687, 0.39215687, 1, '0307_XB.png'),
(320, 0.39215684, 0.3920434, 0.39155868, 1, '0308_XB.png'),
(321, 0.3920022, 0.39187843, 0.39001176, 1, '0309_XB.png'),
(322, 0.39199185, 0.39183718, 0.39105335, 1, '0310_XB.png'),
(323, 0.39190933, 0.39151746, 0.39095023, 1, '0311_XB.png'),
(324, 0.39206406, 0.39166182, 0.39007366, 1, '0312_XB.png'),
(325, 0.39201248, 0.39153808, 0.39066145, 1, '0313_XB.png'),
(326, 0.39207435, 0.3918784, 0.39111528, 1, '0314_XB.png'),
(327, 0.3921053, 0.39201245, 0.39104307, 1, '0315_XB.png'),
(328, 0.39205375, 0.38727874, 0.38882568, 1, '0316_XB.png'),
(329, 0.39198157, 0.39129055, 0.391404, 1, '0317_XB.png'),
(330, 0.3921156, 0.39157933, 0.39041394, 1, '0318_XB.png'),
(331, 0.39198157, 0.39181656, 0.3911668, 1, '0319_XB.png'),
(332, 0.39215687, 0.39190933, 0.39090896, 1, '0320_XB.png'),
(333, 0.39213622, 0.39195064, 0.38828945, 1, '0321_XB.png'),
(334, 0.39204344, 0.39166185, 0.3909399, 1, '0322_XB.png'),
(335, 0.39205375, 0.3917753, 0.39087802, 1, '0323_XB.png'),
(336, 0.3921156, 0.39171344, 0.39002204, 1, '0324_XB.png'),
(337, 0.3920331, 0.3910121, 0.38993955, 1, '0325_XB.png'),
(338, 0.39215687, 0.39213625, 0.39120802, 1, '0326_XB.png'),
(339, 0.3921569, 0.3920847, 0.39194027, 1, '0327_XB.png'),
(340, 0.39206406, 0.39188874, 0.39170307, 1, '0328_XB.png'),
(341, 0.39201248, 0.39183718, 0.39120802, 1, '0329_XB.png'),
(342, 0.3920434, 0.39184743, 0.39143494, 1, '0330_XB.png');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `botol`
--
ALTER TABLE `botol`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `botol`
--
ALTER TABLE `botol`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=343;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
