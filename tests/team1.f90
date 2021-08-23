program team1
implicit none
! Syntax check only(AST)

integer, parameter          :: n = 4
type (team_type)            :: column, odd_even
real,codimension[n, *]      :: co_array
integer,dimension(2)        :: my_cosubscripts
my_cosubscripts (:)   = this_image(co_array)

form team (my_cosubscripts(2), column, new_index = my_cosubscripts(1))
sync team (column)
change team (column, ca[*] => co_array)
! segment 1
end team

formteam (2-mod(this_image(), 2), odd_even)
changeteam (odd_even)
! segment 2
endteam

end program
