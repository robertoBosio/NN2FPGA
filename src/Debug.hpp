#ifndef __DEBUG__
#define __DEBUG__

/* template < */
/* 	class t_input */
/* > void EmptyStream( */
/* 	hls::stream<t_input> &i_data */
/* ) { */

/* #pragma HLS inline */
/* 	/1* This handles padding aware inputs *1/ */

/* #ifndef __SYNTHESIS__ */
/* 	int s_left = 0; */
/* 	while(!i_data.empty()) { */
/* 		i_data.read(); */
/* 		s_left++; */
/* 	} */
/* 	std::cout << "LEFT: " << s_left << std::endl; */
/* #endif */

/* } */

#endif
